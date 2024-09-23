$(document).ready(main);

const example_data = [
    null,
    [{"utility": 20, "agent": "A", "credence": 10, "action": "one"}, {"utility": 10, "agent": "B", "credence": 10, "action": "one"}],
    [{"utility": 20, "agent": "A", "credence": 10, "action": "one"}, {"utility": 0, "agent": "A", "credence": 10, "action": "two"}, {"utility": 0, "agent": "B", "credence": 10, "action": "one"}, {"utility": 10, "agent": "B", "credence": 10, "action": "two"}],
    [{"utility": 20, "agent": "A", "credence": 10, "action": "one"}, {"utility": 5, "agent": "A", "credence": 10, "action": "two"}, {"utility": 5, "agent": "B", "credence": 10, "action": "one"}, {"utility": 10, "agent": "B", "credence": 10, "action": "two"}],
    [{"utility": 20, "agent": "A", "credence": 20, "action": "one"}, {"utility": 0, "agent": "A", "credence": 20, "action": "two"}, {"utility": 0, "agent": "B", "credence": 10, "action": "one"}, {"utility": 10, "agent": "B", "credence": 10, "action": "two"}],
    [{"utility": 20, "agent": "A", "credence": 30, "action": "one"}, {"utility": 5, "agent": "A", "credence": 30, "action": "two"}, {"utility": 5, "agent": "B", "credence": 10, "action": "one"}, {"utility": 10, "agent": "B", "credence": 10, "action": "two"}],
    [{"utility": 10, "agent": "A", "credence": 1, "action": "one"}, {"utility": 10, "agent": "B", "credence": 1, "action": "one"}, {"utility": 10, "agent": "C", "credence": 1, "action": "one"}, {"utility": 10, "agent": "A", "credence": 1, "action": "two"}, {"utility": 10, "agent": "B", "credence": 1, "action": "two"}, {"utility": 10, "agent": "C", "credence": 1, "action": "two"}],
    [{"utility": 10, "agent": "A", "credence": 1, "action": "one"}, {"utility": 10, "agent": "B", "credence": 1, "action": "one"}, {"utility": 10, "agent": "C", "credence": 1, "action": "one"}, {"utility": 10, "agent": "A", "credence": 1, "action": "two"}, {"utility": 20, "agent": "B", "credence": 1, "action": "two"}, {"utility": 30, "agent": "C", "credence": 1, "action": "two"}],
    [{"utility": 1000.0, "agent": "A", "credence": 30, "action": "one"}, {"utility": 1000.0, "agent": "B", "credence": 10, "action": "one"}, {"utility": 1000.0, "agent": "C", "credence": 20, "action": "one"}, {"utility": 1.0, "agent": "A", "credence": 30, "action": "two"}, {"utility": 1.0, "agent": "B", "credence": 10, "action": "two"}, {"utility": 1.0, "agent": "C", "credence": 20, "action": "two"}, {"utility": 1.0, "agent": "A", "credence": 30, "action": "three"}, {"utility": 1.0, "agent": "B", "credence": 10, "action": "three"}, {"utility": 10000.0, "agent": "C", "credence": 20, "action": "three"}],
    [{"utility": 1, "agent": "", "credence": 10, "action": ""}, {"utility": 1, "agent": "elephant", "credence": 10, "action": ""}, {"utility": 1, "agent": "", "credence": 10, "action": ""}],
    [{"utility": 100, "agent": "A", "credence": 10, "action": "one"}, {"utility": 100, "agent": "B", "credence": 10, "action": "one"}, {"utility": 100, "agent": "C", "credence": 10, "action": "one"}, {"utility": 10, "agent": "A", "credence": 10, "action": "two"}, {"utility": 10, "agent": "B", "credence": 10, "action": "two"}, {"utility": 10, "agent": "C", "credence": 10, "action": "two"}],
    [{"utility": 100, "agent": "A", "credence": 10, "action": "one"}, {"utility": 1, "agent": "B", "credence": 10, "action": "one"}, {"utility": 1, "agent": "C", "credence": 10, "action": "one"}, {"utility": 10, "agent": "A", "credence": 10, "action": "two"}, {"utility": 10, "agent": "B", "credence": 10, "action": "two"}, {"utility": 10, "agent": "C", "credence": 10, "action": "two"}],
    [{"utility": 100.0, "agent": "A", "credence": 1, "action": "one"}, {"utility": 100.0, "agent": "B", "credence": 1, "action": "one"}, {"utility": 1.0, "agent": "C", "credence": 1, "action": "one"}, {"utility": 1.0, "agent": "A", "credence": 1, "action": "two"}, {"utility": 1.0, "agent": "B", "credence": 1, "action": "two"}, {"utility": 101.0, "agent": "C", "credence": 1, "action": "two"}],
    [{"utility": 100.0, "agent": "A", "credence": 1, "action": "one"}, {"utility": 100.0, "agent": "B", "credence": 1, "action": "one"}, {"utility": 1.0, "agent": "C", "credence": 2, "action": "one"}, {"utility": 1.0, "agent": "A", "credence": 1, "action": "two"}, {"utility": 1.0, "agent": "B", "credence": 1, "action": "two"}, {"utility": 101.0, "agent": "C", "credence": 2, "action": "two"}]
    ];

const color_scheme = d3.schemeCategory10;
const chart_funcs = {'volume' : volume_chart, 'area' : area_chart, 'both' : both_charts};

const dataURI = $("#variables").attr("src");
let variables;
// const variables = JSON.parse($("#variables").html());
let chart_func = null;
let maximize;
let use_charts;
if (dataURI.startsWith("data:application/json;base64")) {
    
    variables = JSON.parse(atob(dataURI.substring(29)));
    use_charts = variables['chart_type'] !== 'none';
    if (use_charts) {
        chart_func = chart_funcs[variables['chart_type']];
    }
    maximize = variables['maximize'];
} else {
    d3.json(dataURI, function(json) {
        variables = json;
        use_charts = variables['chart_type'] !== 'none';
        if (use_charts) {
            chart_func = chart_funcs[variables['chart_type']];
        }
        maximize = variables['maximize'];
    });
}

// 3 -- health or not

// 4 -- constant proportions or not

const valence_default_min = maximize ? "decrease" : "increase";
const valence_default_max = maximize ? "increase" : "decrease";

let task_description = `<p>In this task we assess how to compromise between different views.</p>
<p>Tell us which of the options specified is <strong>the best compromise</strong> for the given situation.</p>
<p>Whether one option is a better compromise than another is up to you. It might be that multiple parties have to accept a slightly worse outcome for themselves in order to best balance the desires of the group.</p>
<p>In this version of the task, groups <strong>` + (maximize ? "prefer" : "dislike") + ` higher outcomes</strong>.
This means that higher outcomes are ` + (maximize ? "better" : "worse" ) + `.</p>`;
if (use_charts) {
    task_description = task_description +
    `<p>The charts shown might aid your reasoning about the proposals, but they do not contain an obvious answer like in the qualification task.
    We've included answers to those examples.
    </p>`;
}

const qualURI = $("#qualification_answers").attr("src");
let qual_answers;
if (qualURI.startsWith("data:application/json;base64")) {
    qual_answers = JSON.parse(atob(qualURI.substring(29)));
} else {
    d3.json(qualURI, function(json) {
        qual_answers = json;
    });
}

// From: https://developer.mozilla.org/en-US/docs/Glossary/Base64
function base64ToBytes(base64) {
  const binString = atob(base64);
  return Uint8Array.from(binString, (m) => m.codePointAt(0));
}

// From: https://developer.mozilla.org/en-US/docs/Glossary/Base64
function bytesToBase64(bytes) {
  const binString = Array.from(bytes, (x) => String.fromCodePoint(x)).join("");
  return btoa(binString);
}

function get_data(selector) {
    const id = $(selector).attr("id") + "-values";
    const str = $("#" + id).val();
    return bytesToJson(str);
}

function bytesToJson(bytes) {
    return eval(new TextDecoder().decode(base64ToBytes(bytes)));
}

function get_example_data(id) {
    const parts = id.split("-");
    const example_num = Number(parts[parts.length - 1]) - 1;
    return example_data[example_num];
}

function main() {

    $("#qualificationButton").on("click", submit_qual);

    $("#task-description").each(function() {
        $(this).append($("<div>").html(task_description));
    });

    $('.question-data.wait-times').each(function() {
        const wait_times = 'will <strong>' + valence_default_min + `</strong> the average number of days
            a group member must <strong>wait for an appointment</strong>`;
        make_question(this, wait_times, 'days');
    });

    $('.question-data.travel-times').each(function() {
        const travel_times = 'will <strong>' + valence_default_min + `</strong> the average number of minutes
            a group member must <strong>travel for an appointment</strong>`;
        make_question(this, travel_times, 'days');
    });

    $('.question-data.medical-costs').each(function() {
        const medical_costs = 'will <strong>' + valence_default_min + `</strong> the <strong>average cost
            of a medical visit</strong> for each group`;
        make_question(this, medical_costs, 'dollars');
    });

    $('.question-data.life-expectancy').each(function() {
        const life_expectancy = 'will <strong>' + valence_default_max + `</strong> the average number of
            <strong>years a group member will live</strong>`;
        make_question(this, life_expectancy, 'years');
    });

    $('.area-chart').each(function() {
        const width = 540;
        const height = 300;
        const scale = 10;
        const selector = $(this).attr("id");        
        const data = get_example_data(selector);
        const color = get_color(data)
        const chart = make_area_chart(data, color, width, height);
        $(this).append(chart.node());
    });

    $('.volume-chart').each(function() {
        const width = 500;
        const height = 360;
        const scale = $(this).attr('data-scale');
        const selector = $(this).attr("id");        
        const data = get_example_data(selector);
        const color = get_color(data)
        const chart = make_volume_chart(data, color, width, height, scale);
        $(this).append(chart.node());
    });

    $("#qualificationButton").on("click", submit_qual);

    select_qual_answers();
}

function submit_qual() {
    let correct = 0;
    for (const [key, answer] of Object.entries(qual_answers)) {
        let response;
        if (key == "q_question-3D-0") {
            response =  $('input[name=' + key + ']').val();
        } else {
            response = $('input[name=' + key + ']:checked').val();
        }
        if (response == answer) {
            correct++;
        }
    }

    $("#qualificationButton").prop("disabled", true);
    $(".qualification-input").prop("disabled", true);
    // add a secret input to the form here as to whether they were correct or not
    // call fail_qual or pass_qual

    if (correct == Object.keys(qual_answers).length) {
        pass_qual();
    } else {
        fail_qual();
    }
}

function fail_qual() {
    // submit the form!
    $("form[name=mturk_form]").submit();
}

function pass_qual() {
    $("#surveyButton").prop("disabled", false);
    $("#surveyInput").show();
}

function select_qual_answers() {
    // TODO: also trigger this when the qualification task is passed?
    $("#examples input").prop("disabled", "disabled");
    $("#examples #q_question-stacked-1_option_one").prop("checked", "checked");
    $("#examples #q_question-stacked-2_option_one").prop("checked", "checked");
    $("#examples #q_question-stacked-3_option_two").prop("checked", "checked");
    $("#examples #q_question-stacked-4_option_two").prop("checked", "checked");
    $("#examples #q_question-stacked-5_option_one").prop("checked", "checked");
    $("#examples #q_question-stacked-6_option_three").prop("checked", "checked");
    $("#examples #q_question-3D-0_option_one").prop("value", "elephant");
    $("#examples #q_question-3D-1_option_two").prop("checked", "checked");
    $("#examples #q_question-3D-2_option_one").prop("checked", "checked");
    $("#examples #q_question-3D-3_option_two").prop("checked", "checked");
    $("#examples #q_question-3D-4_option_one").prop("checked", "checked");
    $("#examples #q_question-3D-5_option_one").prop("checked", "checked");
    $("#examples #q_question-3D-6_option_one").prop("checked", "checked");
}

function area_chart(data, color, width, height) {
    return $("<div>").attr('class', "pt-4 col-12")
                     .append(($("<p>")
                         .attr('style', "text-align: center")
                         .text(`Stacked Bar Chart`)))
              .append(make_area_chart(data, color, width, height).node());
}

function volume_chart(data, color, width, height) {
    return $("<div>").attr('class', "pt-4 col-12")
                     .append(($("<p>")
                         .attr('style', "text-align: center")
                         .text(`3D Bar Chart`)))
              .append(make_volume_chart(data, color, width, height).node());
}

function both_charts(data, color, width, height) {
    const both = [area_chart(data, color, width, height),
        volume_chart(data, color, width, height)];
    // Not that random, but random enough
    both.sort(() => .5 - Math.random());
    return $("<div>").attr('class', "row")
                     .append(both[0])
                     .append(both[1]);
}

function get_color(data) {
    const scenarios = d3.union(data.map((d) => d.action));

    const color = d3.scaleOrdinal().range(color_scheme)
        .domain(scenarios)
        .range(color_scheme);

    return color;
}

function make_question(element, context, unit) {
    var data;
    try {
        data = get_data(element);
    } catch (e) {
        return
    }
    const selector = $(element).attr("id");

    const width = 900;
    const height = 500;
   
    const scenarios = d3.union(data.map((d) => d.action));
    const sizes = d3.rollup(data, (v) => d3.mean(v, (d) => d.credence), (d) => d.agent);
    const outcomes = d3.index(data, (d) => d.action, (d) => d.agent);
    const all_outcomes = d3.union(data.map((d) => d.utility));


    const color = d3.scaleOrdinal().range(color_scheme)
        .domain(scenarios)
        .range(color_scheme);

    const description = make_question_description(sizes, outcomes, scenarios, color,
                                                  selector, context, unit);

    const q_input = make_question_input(scenarios, color, selector);

    const attn = make_question_attention_check(sizes, outcomes, all_outcomes, scenarios, color, selector);


    let scenario_and_chart =  $("<div>")
                                .attr('class', 'row')
                                .append($("<div>")
                                         .attr('class', 'col-md-12 col-lg-4 ')
                                         .append(description)
                                        );
    const right_side_class = 'col-md-12 col-lg-8';
    let question_class = right_side_class;
    let question_container = $("<div>");
    if (use_charts){
        const chart = chart_func(data, color, width, height);
        scenario_and_chart.append($("<div>")
                                  .attr('class', right_side_class)
                                  .append(chart)
                          );
        question_class = 'col col-10 offset-1';
        question_container = question_container.attr('class', 'row');
    }

    const questions = $('<div>')
                         .attr('class', question_class)
                         .append(attn)
                         .append($("<hr />"))
                         .append(q_input);

    if (use_charts) {
        $('#' + selector).append(scenario_and_chart)
                         .append(question_container.append(questions));

        

    } else {

        $('#' + selector).append(scenario_and_chart.append(questions));
    }
}

function make_question_description(sizes, outcomes, scenarios, color,
                                   q_num, context, unit) {

    const groups = Array.from(sizes.keys());

    let group_description = $('<p>')
                             .text('In this scenario, there are ' + groups.length + ' groups:');
    let group_list = $('<ul>');
    for (let i = 0; i < groups.length; i++) {
        let group = groups[i];
        let text = '<span class=group-'.concat(group, '>group ', group, '</span>',
                   ' with <strong>',sizes.get(group), '</strong> people in it');
        if (i < groups.length - 2) {
            text += ", ";
        } else if (i < groups.length - 1) {
            text += ", and ";
        } else {
            text += ".";
        }
        group_list.append($('<li>')
                            .html(text)
                         );
    }

    let proposals_description = 'There are '.concat(scenarios.size, 
                                ' proposals, each of which ', context, ' by:')
    let proposals_list = $("<ul>");

    scenarios.forEach(function(proposal) {
        // Legend squares
        const dot = d3.create('svg')
                      .attr("width", 15)
                      .attr("height", 15);
        dot.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", 15)
            .attr("height", 15)
            .style("fill", color(proposal));
        const square = $("<span>").append(dot.node());

        let scen_text = 'proposal <span class="font-weight-bold"'.concat('style="color : ',
                        color(proposal), '">', proposal, '</span> ', square.html(), ': ');
        for (let i = 0; i < groups.length; i++) {
            let group = groups[i];
            let utility = outcomes.get(proposal).get(group).utility;
            let text = utility.toString() + " " + unit + ' for group ' + group;
            if (i < groups.length - 2) {
                text += ", ";
            } else if (i < groups.length - 1) {
                text += ", and ";
            } else {
                text += ".";
            }
            scen_text += text;
        }

        proposals_list.append($("<li>")
                      .html(scen_text));

    });

    return $("<div>")
     .append(group_description)
     .append(group_list)
     .append($("<p>")
             .html(proposals_description)
            )
     .append(proposals_list);
}

function make_question_attention_check(sizes, outcomes, all_outcomes, scenarios, color, q_num) {
    // randomly choose size or outcome
    // randomly choose an agent
    // if outcome randomly choose an action

    let show_size = Math.random() < .5 ? true : false;
    const equal_groups = (new Set(sizes.values())).size == 1;
    if (equal_groups) {
        show_size = false;
    }
    const chosen_group  = shuffle(Array.from(sizes.keys()))[0];
    const chosen_proposal = shuffle(Array.from(outcomes.keys()))[0];

    let html = 'What is the size of group ' + chosen_group + '?';
    let answer = sizes.get(chosen_group);
    let options = Array.from(sizes.values());
    if (show_size === false) {
        let prop = '<span class="font-weight-bold" style="color : '.concat(color(chosen_proposal),
                                                                           ';">', chosen_proposal, '</span>');
        html = 'What is the outcome for group '.concat(chosen_group, ' of proposal ',
                                                       prop, '?');
        answer = outcomes.get(chosen_proposal).get(chosen_group).utility;
        options = all_outcomes;
    }

    var outside_div = $("<div>")
                    .attr('class', 'row')
                    .append($("<div>")
                             .attr('class', 'offset-md-1 col col-md-5 col-sm-12 pt-4')
                             .append($("<p>")
                                    .html(html)
                                    )
                            )
                    .append($("<input>")
                             .attr('name', 'q_' + q_num + '_attn_answer')
                             .attr('id', 'q_' + q_num + '_attn' + '_answer')
                             .attr('value', answer)
                             .attr('hidden', true));

    var answer_div = $("<div>")
                    .attr('class', 'mt-2 col-md-5 col-sm-12');

    // need to add hidden input for answer


    options.forEach(function(option) {
        answer_div.append($("<div>")
                          .attr('class', 'col-12')
                          .append($("<input>")
                                    .attr('type', 'radio')
                                    .attr('name', 'q_' + q_num + '_attn')
                                    .attr('id', 'q_' + q_num + '_attn' + '_opt_' + option)
                                    .attr('value', option)
                                    .prop('required', true)
                                 )
                          .append($("<label>")
                                    .text(option)
                                 )
                          );
    });

    outside_div.append(answer_div);

    return outside_div;
}

function make_question_input(proposals, color, q_num) {

    var outside_div = $("<div>")
                        .attr('class', 'row')
                        .append($("<div>")
                                 .attr('class', 'offset-md-1 col col-md-5 col-sm-12 pt-4')
                                 .append($("<p>")
                                        .text("Which proposal is the best compromise in this situation?")
                                        )
                                );

    var prop_div = $("<div>")
                    .attr('class', 'mt-2 col-md-5 col-sm-12');

    proposals.forEach(function(proposal) {
        const span = 'Proposal <span class="font-weight-bold" style="color : '.concat(
                     color(proposal), ';">', proposal, '</span>');

        prop_div.append($("<div>")
                          .attr('class', 'col-12')
                          .append($("<input>")
                                    .attr('type', 'radio')
                                    .attr('name', 'q_' + q_num)
                                    .attr('id', 'q_' + q_num + '_proposal_' + proposal)
                                    .attr('value', proposal)
                                    .prop('required',true)
                                 )
                          .append($("<label>")
                                    .html(span)
                                 )
                          );
    });

    outside_div.append(prop_div);

    return outside_div;
}

// from : https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-arrays
function shuffle(unshuffled) {
    let shuffled = unshuffled
        .map(value => ({ value, sort: Math.random() }))
        .sort((a, b) => a.sort - b.sort)
        .map(({ value }) => value);
    return shuffled;
}
