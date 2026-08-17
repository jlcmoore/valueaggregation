#' Mixed-effects analysis using brms for preregistered multinomial models.
#'
#' Fits either the disagree or agree model and reports posterior contrasts.

#' Load required packages quietly.
suppress_pkg_startup <- function() {
  suppressPackageStartupMessages(library(brms))
  suppressPackageStartupMessages(library(ggplot2))
}

chain_count <- 4L
core_count <- 4L
iter_count <- 2000L
warmup_count <- 1000L
adapt_delta <- 0.95
alpha_min <- 0
alpha_max <- 1
alpha_grid_size <- 101L
interval_level <- 0.95
seed_value <- 1234L

#' Require a package or stop with a clear error.
require_package <- function(pkg_name) {
  # Check for required packages before running the analysis.
  if (!requireNamespace(pkg_name, quietly = TRUE)) {
    stop("Missing required package: ", pkg_name, call. = FALSE)
  }
}

#' Parse a comma-separated list into a vector.
parse_csv_list <- function(value) {
  # Split comma-separated arguments into a character vector.
  if (is.null(value)) {
    return(NULL)
  }
  trimws(strsplit(value, ",", fixed = TRUE)[[1]])
}

#' Parse command line arguments for the analysis.
parse_args <- function() {
  # Parse command line arguments for analysis configuration.
  args <- commandArgs(trailingOnly = TRUE)
  res <- list(
    data = NULL,
    analysis = NULL,
    choice_col = "choice",
    condition_col = "condition",
    scenario_type_col = "scenario_type",
    alpha_col = "alpha",
    participant_id_col = "participant_id",
    scenario_id_col = "scenario_id",
    filter_col = NULL,
    filter_value = NULL,
    choice_levels = NULL,
    nash_label = "Nash",
    ie_label = "IE",
    decoy_label = "Decoy",
    nash_ie_label = "Nash/IE",
    by_condition_spline = FALSE,
    out_dir = "data/analysis",
    plot_file = NULL,
    fit_file = NULL,
    load_fit_file = NULL
  )

  known_args <- list(
    "--data" = "data",
    "--analysis" = "analysis",
    "--choice-col" = "choice_col",
    "--condition-col" = "condition_col",
    "--scenario-type-col" = "scenario_type_col",
    "--alpha-col" = "alpha_col",
    "--participant-id-col" = "participant_id_col",
    "--scenario-id-col" = "scenario_id_col",
    "--filter-col" = "filter_col",
    "--filter-value" = "filter_value",
    "--choice-levels" = "choice_levels",
    "--nash-label" = "nash_label",
    "--ie-label" = "ie_label",
    "--decoy-label" = "decoy_label",
    "--nash-ie-label" = "nash_ie_label",
    "--by-condition-spline" = "by_condition_spline",
    "--out-dir" = "out_dir",
    "--plot-file" = "plot_file",
    "--fit-file" = "fit_file",
    "--load-fit" = "load_fit_file"
  )

  idx <- 1
  while (idx <= length(args)) {
    key <- args[[idx]]
    if (key == "--by-condition-spline") {
      res$by_condition_spline <- TRUE
      idx <- idx + 1
    } else if (key %in% names(known_args) && idx + 1 <= length(args)) {
      field_name <- known_args[[key]]
      res[[field_name]] <- args[[idx + 1]]
      idx <- idx + 2
    } else {
      idx <- idx + 1
    }
  }

  if (!is.null(res$choice_levels)) {
    res$choice_levels <- parse_csv_list(res$choice_levels)
  }

  if (is.null(res$data)) {
    stop("Must supply --data <path-to-csv>.", call. = FALSE)
  }
  if (is.null(res$analysis) || !res$analysis %in% c("disagree", "agree")) {
    stop("--analysis must be 'disagree' or 'agree'.", call. = FALSE)
  }

  if (is.null(res$plot_file)) {
    res$plot_file <- paste0("alpha_probabilities_", res$analysis, ".csv")
  }
  if (is.null(res$fit_file)) {
    res$fit_file <- paste0("brms_fit_", res$analysis, ".rds")
  }

  res
}

#' Prepare and validate data for modeling.
prepare_data <- function(dat, opts) {
  # Validate columns, apply optional filters, and coerce key columns.
  required_cols <- c(
    opts$choice_col,
    opts$condition_col,
    opts$scenario_type_col,
    opts$participant_id_col,
    opts$scenario_id_col
  )
  if (opts$analysis == "disagree") {
    required_cols <- c(required_cols, opts$alpha_col)
  }
  missing_cols <- setdiff(required_cols, names(dat))
  if (length(missing_cols) > 0L) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "),
         call. = FALSE)
  }

  if (!is.null(opts$filter_col)) {
    if (!(opts$filter_col %in% names(dat))) {
      stop("Filter column not found: ", opts$filter_col, call. = FALSE)
    }
    if (is.null(opts$filter_value)) {
      stop("Must supply --filter-value when using --filter-col.", call. = FALSE)
    }
    dat <- dat[dat[[opts$filter_col]] == opts$filter_value, , drop = FALSE]
  }

  dat[[opts$condition_col]] <- factor(dat[[opts$condition_col]])
  dat[[opts$scenario_type_col]] <- factor(dat[[opts$scenario_type_col]])
  dat[[opts$participant_id_col]] <- factor(dat[[opts$participant_id_col]])
  dat[[opts$scenario_id_col]] <- factor(dat[[opts$scenario_id_col]])

  if (opts$analysis == "disagree") {
    dat[[opts$alpha_col]] <- as.numeric(dat[[opts$alpha_col]])
  }

  choice_levels <- opts$choice_levels
  if (is.null(choice_levels)) {
    if (opts$analysis == "disagree") {
      choice_levels <- c(opts$decoy_label, opts$ie_label, opts$nash_label)
    } else {
      choice_levels <- c(opts$decoy_label, opts$nash_ie_label)
    }
  }
  if (!all(choice_levels %in% unique(dat[[opts$choice_col]]))) {
    stop(
      "Choice levels not found in data: ",
      paste(setdiff(choice_levels, unique(dat[[opts$choice_col]])),
            collapse = ", "),
      call. = FALSE
    )
  }
  dat[[opts$choice_col]] <- factor(dat[[opts$choice_col]],
                                   levels = choice_levels)

  droplevels(dat)
}

#' Build priors for brms.
build_priors <- function() {
  NULL
}

#' Ensure the output directory exists.
ensure_output_dir <- function(out_dir) {
  # Create the output directory if needed before writing files.
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
  }
}

#' Fit the categorical model for disagreement trials.
fit_disagree_model <- function(dat, opts) {
  # Fit the categorical model for disagreement trials.
  # Optional condition-specific crossover:
  #   choice ~ condition + scenario_type + s(alpha) + s(alpha, by = condition)
  #     + (1 | participant_id) + (1 | scenario_id)
  # This keeps a shared smooth while letting each condition deviate in shape.
  alpha_term <- paste0("s(", opts$alpha_col, ")")
  if (opts$by_condition_spline) {
    alpha_term <- paste0(
      alpha_term,
      " + s(",
      opts$alpha_col,
      ", by = ",
      opts$condition_col,
      ")"
    )
  }
  form <- as.formula(paste(
    opts$choice_col,
    "~",
    paste(
      "1 +",
      opts$condition_col,
      "+",
      opts$scenario_type_col,
      "+",
      alpha_term,
      "+ (1 |", opts$participant_id_col, ")",
      "+ (1 |", opts$scenario_id_col, ")"
    )
  ))

  brm(
    formula = form,
    data = dat,
    family = categorical(),
    prior = build_priors(),
    cores = core_count,
    chains = chain_count,
    iter = iter_count,
    warmup = warmup_count,
    seed = seed_value,
    control = list(adapt_delta = adapt_delta),
    refresh = 0
  )
}

#' Fit the categorical model for agreement trials.
fit_agree_model <- function(dat, opts) {
  # Fit the categorical model for agreement trials.
  form <- as.formula(paste(
    opts$choice_col,
    "~",
    paste(
      "1 +",
      opts$condition_col,
      "+",
      opts$scenario_type_col,
      "+ (1 |", opts$participant_id_col, ")",
      "+ (1 |", opts$scenario_id_col, ")"
    )
  ))

  brm(
    formula = form,
    data = dat,
    family = categorical(),
    prior = build_priors(),
    cores = core_count,
    chains = chain_count,
    iter = iter_count,
    warmup = warmup_count,
    seed = seed_value,
    control = list(adapt_delta = adapt_delta),
    refresh = 0
  )
}

#' Create a fixed alpha grid in [0, 1] for predictions.
make_alpha_grid <- function() {
  # Create a fixed alpha grid in [0, 1] for posterior predictions.
  seq(alpha_min, alpha_max, length.out = alpha_grid_size)
}

#' Summarize posterior draws with mean and interval bounds.
summarize_probabilities <- function(draws) {
  # Summarize posterior draws with mean and interval bounds.
  lower_q <- (1 - interval_level) / 2
  upper_q <- 1 - lower_q
  draws_vec <- as.numeric(draws)
  data.frame(
    mean = mean(draws_vec),
    lower = quantile(draws_vec, probs = lower_q),
    upper = quantile(draws_vec, probs = upper_q)
  )
}

#' Compute posterior contrasts for disagree trials and save plot data.
compute_disagree_contrasts <- function(fit, dat, opts) {
  # Compute H1/H2 posterior contrasts and a probability plot.
  alpha_grid <- make_alpha_grid()
  condition_levels <- levels(dat[[opts$condition_col]])
  scenario_levels <- levels(dat[[opts$scenario_type_col]])

  grid <- expand.grid(
    condition = condition_levels,
    scenario_type = scenario_levels,
    alpha = alpha_grid,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  names(grid) <- c(opts$condition_col, opts$scenario_type_col, opts$alpha_col)

  posterior <- posterior_epred(
    fit,
    newdata = grid,
    re_formula = NA,
    allow_new_levels = TRUE
  )

  category_levels <- dimnames(posterior)[[3]]
  nash_idx <- match(opts$nash_label, category_levels)
  ie_idx <- match(opts$ie_label, category_levels)
  decoy_idx <- match(opts$decoy_label, category_levels)

  if (any(is.na(c(nash_idx, ie_idx, decoy_idx)))) {
    stop("Could not locate expected choice levels in posterior output.",
         call. = FALSE)
  }

  cat("\n--- Disagree trials: posterior contrasts ---\n")
  for (cond_level in condition_levels) {
    row_idx <- grid[[opts$condition_col]] == cond_level
    avg_probs <- apply(posterior[, row_idx, , drop = FALSE], c(1, 3), mean)

    prob_h1 <- mean(avg_probs[, nash_idx] > avg_probs[, decoy_idx])
    prob_h2 <- mean(avg_probs[, nash_idx] > avg_probs[, ie_idx])

    cat("\nCondition:", cond_level, "\n")
    cat("  H1: P(Nash) > P(Decoy) =", sprintf("%.3f", prob_h1), "\n")
    cat("  H2: P(Nash) > P(IE)    =", sprintf("%.3f", prob_h2), "\n")

    alpha_matches <- grid[[opts$condition_col]] == cond_level
    alpha_values <- sort(unique(grid[[opts$alpha_col]]))
    dominance_by_alpha <- matrix(FALSE, nrow = nrow(avg_probs),
                                 ncol = length(alpha_values))

    for (alpha_idx in seq_along(alpha_values)) {
      alpha_value <- alpha_values[[alpha_idx]]
      row_alpha <- alpha_matches & grid[[opts$alpha_col]] == alpha_value
      alpha_probs <- apply(
        posterior[, row_alpha, , drop = FALSE],
        c(1, 3),
        mean
      )
      dominance_by_alpha[, alpha_idx] <- alpha_probs[, nash_idx] >
        alpha_probs[, ie_idx]
    }
    prob_global <- mean(apply(dominance_by_alpha, 1, all))
    cat("  H2 (all alpha): P(Nash > IE across alpha) =",
        sprintf("%.3f", prob_global), "\n")
  }

  plot_rows <- list()
  for (cond_level in condition_levels) {
    alpha_values <- sort(unique(grid[[opts$alpha_col]]))
    for (alpha_value in alpha_values) {
      row_alpha <- grid[[opts$condition_col]] == cond_level &
        grid[[opts$alpha_col]] == alpha_value
      alpha_probs <- apply(
        posterior[, row_alpha, , drop = FALSE],
        c(1, 3),
        mean
      )
      nash_summary <- summarize_probabilities(
        alpha_probs[, nash_idx, drop = FALSE]
      )
      ie_summary <- summarize_probabilities(
        alpha_probs[, ie_idx, drop = FALSE]
      )

      plot_rows[[length(plot_rows) + 1L]] <- data.frame(
        condition = cond_level,
        alpha = alpha_value,
        choice = opts$nash_label,
        mean = nash_summary$mean,
        lower = nash_summary$lower,
        upper = nash_summary$upper
      )
      plot_rows[[length(plot_rows) + 1L]] <- data.frame(
        condition = cond_level,
        alpha = alpha_value,
        choice = opts$ie_label,
        mean = ie_summary$mean,
        lower = ie_summary$lower,
        upper = ie_summary$upper
      )
    }
  }

  plot_data <- do.call(rbind, plot_rows)
  plot_path <- file.path(opts$out_dir, opts$plot_file)
  write.csv(plot_data, plot_path, row.names = FALSE)
  cat("\nSaved alpha probability data to:", plot_path, "\n")
}

#' Compute posterior contrasts for agree trials.
compute_agree_contrasts <- function(fit, dat, opts) {
  # Compute H3 posterior contrasts for agree trials.
  condition_levels <- levels(dat[[opts$condition_col]])
  scenario_levels <- levels(dat[[opts$scenario_type_col]])

  grid <- expand.grid(
    condition = condition_levels,
    scenario_type = scenario_levels,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  names(grid) <- c(opts$condition_col, opts$scenario_type_col)

  posterior <- posterior_epred(
    fit,
    newdata = grid,
    re_formula = NA,
    allow_new_levels = TRUE
  )

  category_levels <- dimnames(posterior)[[3]]
  target_idx <- match(opts$nash_ie_label, category_levels)
  if (is.na(target_idx)) {
    stop("Could not locate Nash/IE label in posterior output.",
         call. = FALSE)
  }

  cat("\n--- Agree trials: posterior contrasts ---\n")
  plot_rows <- list()
  for (cond_level in condition_levels) {
    row_idx <- grid[[opts$condition_col]] == cond_level
    avg_probs <- apply(posterior[, row_idx, , drop = FALSE], c(1, 3), mean)
    prob_h3 <- mean(avg_probs[, target_idx] > (1 / 3))
    cat("\nCondition:", cond_level, "\n")
    cat("  H3: P(Nash/IE) > 1/3 =", sprintf("%.3f", prob_h3), "\n")

    for (choice_idx in seq_len(ncol(avg_probs))) {
      summary_row <- summarize_probabilities(avg_probs[, choice_idx])
      plot_rows[[length(plot_rows) + 1L]] <- data.frame(
        condition = cond_level,
        choice = category_levels[[choice_idx]],
        mean = summary_row$mean,
        lower = summary_row$lower,
        upper = summary_row$upper
      )
    }
  }

  plot_data <- do.call(rbind, plot_rows)
  plot_path <- file.path(opts$out_dir, opts$plot_file)
  write.csv(plot_data, plot_path, row.names = FALSE)
  cat("\nSaved agree probability data to:", plot_path, "\n")
}

#' Run the analysis end-to-end.
run_analysis <- function() {
  # Main analysis entry point.
  require_package("brms")
  require_package("cmdstanr")
  require_package("ggplot2")
  suppress_pkg_startup()

  options(brms.backend = "cmdstanr")
  options(mc.cores = core_count)
  set.seed(seed_value)

  opts <- parse_args()
  ensure_output_dir(opts$out_dir)
  dat <- read.csv(opts$data, stringsAsFactors = FALSE)
  dat <- prepare_data(dat, opts)

  if (opts$analysis == "disagree") {
    if (!is.null(opts$load_fit_file)) {
      fit <- readRDS(opts$load_fit_file)
    } else {
      fit <- fit_disagree_model(dat, opts)
      saveRDS(fit, file.path(opts$out_dir, opts$fit_file))
    }
    print(summary(fit))
    compute_disagree_contrasts(fit, dat, opts)
  } else {
    if (!is.null(opts$load_fit_file)) {
      fit <- readRDS(opts$load_fit_file)
    } else {
      fit <- fit_agree_model(dat, opts)
      saveRDS(fit, file.path(opts$out_dir, opts$fit_file))
    }
    print(summary(fit))
    compute_agree_contrasts(fit, dat, opts)
  }
}

run_analysis()
