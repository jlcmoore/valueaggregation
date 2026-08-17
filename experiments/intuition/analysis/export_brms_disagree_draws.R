#' Export posterior draw probabilities for disagree trials.
#'
#' Usage:
#'   Rscript export_brms_disagree_draws.R --fit <fit.rds> --out <output.csv>

suppress_pkg_startup <- function() {
  suppressPackageStartupMessages(library(brms))
}

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  res <- list(
    fit = NULL,
    out = NULL,
    condition_col = "condition",
    scenario_type_col = "scenario_type",
    alpha_col = "alpha",
    nash_label = "Nash",
    ie_label = "IE"
  )
  known_args <- list(
    "--fit" = "fit",
    "--out" = "out",
    "--condition-col" = "condition_col",
    "--scenario-type-col" = "scenario_type_col",
    "--alpha-col" = "alpha_col",
    "--nash-label" = "nash_label",
    "--ie-label" = "ie_label"
  )

  idx <- 1
  while (idx <= length(args)) {
    key <- args[[idx]]
    if (key %in% names(known_args) && idx + 1 <= length(args)) {
      field_name <- known_args[[key]]
      res[[field_name]] <- args[[idx + 1]]
      idx <- idx + 2
    } else {
      idx <- idx + 1
    }
  }

  if (is.null(res$fit) || is.null(res$out)) {
    stop("Usage: --fit <fit.rds> --out <output.csv>", call. = FALSE)
  }
  res
}

ensure_factor <- function(vec) {
  if (is.factor(vec)) {
    return(vec)
  }
  factor(vec)
}

export_draws <- function(opts) {
  fit <- readRDS(opts$fit)
  dat <- fit$data
  if (is.null(dat)) {
    stop("Fit object does not contain data.", call. = FALSE)
  }

  dat[[opts$condition_col]] <- ensure_factor(dat[[opts$condition_col]])
  dat[[opts$scenario_type_col]] <- ensure_factor(dat[[opts$scenario_type_col]])
  dat[[opts$alpha_col]] <- ensure_factor(dat[[opts$alpha_col]])

  condition_levels <- levels(dat[[opts$condition_col]])
  scenario_levels <- levels(dat[[opts$scenario_type_col]])
  alpha_levels <- levels(dat[[opts$alpha_col]])
  if (length(condition_levels) == 0L ||
      length(scenario_levels) == 0L ||
      length(alpha_levels) == 0L) {
    stop("Missing levels for prediction grid.", call. = FALSE)
  }

  grid <- expand.grid(
    condition = condition_levels,
    scenario_type = scenario_levels,
    alpha = alpha_levels,
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
  if (any(is.na(c(nash_idx, ie_idx)))) {
    stop("Could not find Nash/IE labels in posterior output.", call. = FALSE)
  }

  draw_count <- dim(posterior)[[1]]
  draw_rows <- list()
  for (cond_level in condition_levels) {
    for (alpha_value in alpha_levels) {
      row_alpha <- grid[[opts$condition_col]] == cond_level &
        grid[[opts$alpha_col]] == alpha_value
      alpha_probs <- apply(
        posterior[, row_alpha, , drop = FALSE],
        c(1, 3),
        mean
      )
      draw_rows[[length(draw_rows) + 1L]] <- data.frame(
        condition = cond_level,
        alpha = as.numeric(as.character(alpha_value)),
        draw = seq_len(draw_count),
        nash_prob = alpha_probs[, nash_idx],
        ie_prob = alpha_probs[, ie_idx]
      )
    }
  }

  draw_data <- do.call(rbind, draw_rows)
  write.csv(draw_data, opts$out, row.names = FALSE)
  cat("Saved alpha draw data to:", opts$out, "\n")
}

run_export <- function() {
  suppress_pkg_startup()
  opts <- parse_args()
  export_draws(opts)
}

run_export()
