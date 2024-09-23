
function make_legend(domain, color) {
    // Legend
    // from: https://d3-graph-gallery.com/graph/custom_legend.html
    
    const lg_width = 100;
    const lg_height = 75;
    var svg = d3.create("svg")
        .attr("width", lg_width)
        .attr("height", lg_height)
        .attr("viewBox", [0, 0, lg_width, lg_height])
        .attr("style", "max-width: 100%; height: auto;");

    // Legend squares
    var lg_sq_size = 20

    svg.selectAll("lg-dots")
      .data(domain)
      .enter()
      .append("rect")
        .attr("class", "lg-dots")
        .attr("x", 0)
        .attr("y", function(d, i) { return i * (lg_sq_size + 5)})
        // 100 is where the first dot appears. 25 is the distance between dots
        .attr("width", lg_sq_size)
        .attr("height", lg_sq_size)
        .style("fill", (d) => color(d))

    // Legend text
    svg.selectAll("lg-text")
      .data(domain)
      .enter()
      .append("text")
        .attr("class", "lg-text")
        .attr("x", lg_sq_size * 1.2)
        .attr("y", function(d, i) { return i * (lg_sq_size + 5) + (lg_sq_size / 2)})
        // 100 is where the first dot appears. 25 is the distance between dots
        .style("fill", (d) => color(d))
        .text((d) => d)
        .attr("text-anchor", "left")
        .style("alignment-baseline", "middle")

    return svg;
}

function make_area_chart(data, color, width, height) {

    const marginTop = 10;
    const marginRight = 15;
    const marginBottom = 20;
    const marginLeft = 40;
    const group_gap = 5;

    // These are needed to compute the bar width, giving
    // the max size column the full width
    const sizes = d3.rollup(data, (v) => d3.mean(v, (d) => d.credence) , (d) => d.agent);
    const max_size = d3.rollup(data, (v) => d3.max(v, (d) => d.credence));
    const sum_size = d3.rollup(data, (v) => d3.sum(v, (d) => d.credence));

    const scenarios = d3.union(data.map((d) => d.action));

    // Determine the series that need to be stacked.
    const series = d3.stack()
        .keys(scenarios) // distinct series keys, in input order
        .value(([, D], key) => D.get(key).utility) // get value for each series key and stack
        (d3.index(data, (d) => d.agent, (d) => d.action)); // group by stack then series key

    const pie = d3.pie()
        .sortValues(null)
        .value((d) => d[1]);
    
    const arcs = pie(sizes);

    // Re-scale these to be percentages
    arcs.forEach(function(d) {
        d.startAngle = (d.startAngle  / (Math.PI * 2));
        d.endAngle = (d.endAngle  / (Math.PI * 2));
    });
    const xPos = d3.index(arcs, (d) => d.data[0])

    const groups = d3.groupSort(data, (D) => -d3.sum(D, (d) => d.utility), (d) => d.agent)

    const xScale = d3.scaleBand()
        .domain(groups)
        .range([marginLeft, width - marginRight])
        .padding(.01);

    // Prepare the scales for positional and color encodings.
    const widthScale = d3.scaleLinear()
        .domain([0, 1])
        .range([marginLeft, width - marginRight]);

    const yScale = d3.scaleLinear()
        .domain([0, d3.max(series, (d) => d3.max(d, (d) => d[1]))])
        .rangeRound([height - marginBottom, marginTop]);

    // color palette = one color per subgroup (scenarios)
    var svg = d3.create("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .attr("style", "max-width: 100%; height: auto;");

    // Add the bars
    svg.append("g")
        .selectAll()
        .data(series)
        .join("g")
            .attr("fill", (d) => color(d.key))
            // Could add back strokes for bars
            // .style('stroke', 'black')
            // .style("stroke-width", (d) => "1.5")
        .selectAll("rect")
        .data((D) => D.map((d) => (d.key = D.key, d)))
        .join("rect")
            .attr("x", (d) => widthScale(xPos.get(d.data[0]).startAngle))
            .attr("y", (d) => yScale(d[1]))
            .attr("height", (d) => yScale(d[0]) - yScale(d[1]))
            .attr("width", (d) => widthScale(xPos.get(d.data[0]).endAngle) - widthScale(xPos.get(d.data[0]).startAngle))
        .append("title")
            .text((d) => 'group: '.concat(d.data[0], '\nscenario: ', d.key, '\nvalue: ',
                                          d.data[1].get(d.key).utility));
 
    // Append the horizontal axis (percentages)
    const formatPercent = d3.format(".0%");
    svg.append("g")
        .attr("transform", 'translate(0, ' + (height - marginBottom) + ')')
        .call(d3.axisBottom(widthScale).tickSizeOuter(0).tickValues([0,1]).tickFormat(formatPercent))
        .call((g) => g.selectAll(".domain").remove());

    // Append the group label axis.
    svg.selectAll(".group-label")
        .data(xPos)
        .enter()
        .append('text')
        .attr("class", "group-label")
        .attr("text-anchor", "middle")
        .attr('x', (d) => widthScale(d[1].startAngle + ((d[1].endAngle - d[1].startAngle) / 2)))
        .attr('y', yScale(0) + 15)
        .text((d) => d[0]);

    // Gaps between variables
    svg.selectAll(".group-gap")
        .data(series[series.length - 1])
        .enter()
        .append('line')
            .attr("class", "group-gap")
            .style("stroke", "white")
            .style("stroke-width", (d) => group_gap)
            .attr("x1", (d) => widthScale(xPos.get(d.data[0]).startAngle)) 
            .attr("y1", yScale(0) + 1)
            .attr("x2", (d) => widthScale(xPos.get(d.data[0]).startAngle)) 
            .attr("y2", (d) => -(height - marginBottom));
        // +1, -1 here to make sure it covers
        // the rectangles. change if making these
        // other than white

    // Fence post, final bar gap
    svg.append('line')
        .attr("class", "group-gap")
        .style("stroke", "white")
        .style("stroke-width", (d) => group_gap)
        .attr("x1", (d) => widthScale(100)) 
        .attr("y1", yScale(0) - 1)
        .attr("x2", (d) => widthScale(100)) 
        .attr("y2", (d) => -(height - marginBottom));

    // Append the vertical axis (utilities)
    svg.append("g")
        .attr("transform", 'translate( ' + marginLeft + ', 0)')
        .call(d3.axisLeft(yScale).ticks(null, "s"))
        .call((g) => g.selectAll(".domain").remove());

    return svg;
}

function make_volume_chart(o_data, color, width, height, scale = null) {
    // Derived from: https://gist.github.com/Niekes/613d43d39372f99ae2dcea14f0f90617
    // for the cubes
    // and https://gist.github.com/niekes/1c15016ae5b5f11508f92852057136b5
    // for the lines
    console.log(o_data);
    var data = JSON.parse(JSON.stringify(o_data));
    const origin = [75, 150];
    const cubesData = [];
    const yLine = [];
    const xLine = [];
    const zLine = [];
    var alpha = 0;
    var beta = 0;
    const startAngle = Math.PI/6;

    var yScale3d, xScale3d, zScale3d;

    // Set-up cubes

    const sum_size = d3.rollup(data, (v) => d3.sum(v, (d) => d.credence));

    const asymmetric = d3.union(data.map((d) => d.credence)).size != 1;
    if (asymmetric) {
        data = data.map(function(d) {
            d.utility = d.utility ** (d.credence / sum_size);
            return d;
        });
    }

    const grouped = d3.group(data, (d) => d.action);
    var keys = d3.union(data.map((d) => d.action));
    var next_x = 0;
    var max_length = 0;
    var min_length = null;
    
    // Detect whether this is a symmetric or asymmetric case, only
    // use the exponential in the asymmetric case

    const prelim_cubes = {};
    keys.forEach(function (key) {
        const utils = grouped.get(key);
        let utilities;
        if (asymmetric) {
            utilities = d3.map(utils, (v) => v.utility ** (v.credence / sum_size));
        } else {
            utilities = d3.map(utils, (v) => v.utility);
        }

        // TO deal with cases when there are not three agents
        if (utilities.length == 2) {
            utilities.push(0);
        } else if (utilities.length == 1) {
            utilities.push(0);
            utilities.push(0);
        }

        prelim_cubes[key] = utilities;

        this_max_length = Math.max(...utilities); //d3.rollup(utils, (v) => d3.max(v, (d) => d.utility));
        this_min_length = Math.min(...utilities); //d3.rollup(utils, (v) => d3.min(v, (d) => d.utility));
        max_length = Math.max(max_length, this_max_length);
        min_length = Math.min(min_length, this_min_length);
    });

    const new_gap = (max_length - min_length) / 4;

    next_x = 0;

    keys.forEach(function (key) {
        const cube = prelim_cubes[key];
        const x_length = cube[0];
        const y_length = cube[1];
        const z_length = cube[2];

        var _cube = makeCube(next_x, 0, 0,
                             next_x + x_length,
                             y_length,
                             z_length);
        next_x = next_x + new_gap + x_length;

        _cube.id = 'cube_' + key;
        _cube.scenario = key.toString();

        cubesData.push(_cube);
    });


    const set_scale = scale === null;
    // For the scale we want the scale = longest side / min(height, width)

    if (set_scale) {
        // TODO: maybe add margin top to scale...
        // tasks 10, 11, 12
        max_length = Math.max(max_length, next_x);
        // Subtract the origin offset
        scale = Math.min(width - origin[0], height - origin[1]) / max_length;
    }

    var svg = d3.create('svg');
        
    svg.attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .attr("style", "max-width: 100%; height: auto;")
        .call(d3.drag()
                .on('drag', dragged)
                .on('start', dragStart)
                .on('end', dragEnd))
        .append('g');

    var cubesGroup = svg.append('g')
                        .attr('class', 'cubes');
    var mx, my, mouseX, mouseY;


    var cubes3D = d3._3d()
        .shape('CUBE')
        .x((d) => d.x)
        .y((d) => d.y)
        .z((d) => d.z)
        .rotateY( startAngle)
        .rotateX(-startAngle)
        .origin(origin)
        .scale(scale);

    yScale3d = d3._3d()
        .shape('LINE_STRIP')
        .origin(origin)
        .rotateY( startAngle)
        .rotateX(-startAngle)
        .scale(scale);

    xScale3d = d3._3d()
        .shape('LINE_STRIP')
        .origin(origin)
        .rotateY( startAngle)
        .rotateX(-startAngle)
        .scale(scale);

    zScale3d = d3._3d()
        .shape('LINE_STRIP')
        .origin(origin)
        .rotateY( startAngle)
        .rotateX(-startAngle)
        .scale(scale);

    // Set-up axes

    const agents = Array.from(d3.union(data.map((d) => d.agent)));

    const axis_offset = 1 / 10;
    const x_origin = (next_x - new_gap) + (next_x - new_gap) * axis_offset; // -gap because we need to fencepost     
    const axis_length = Math.max(x_origin, max_length);

    const start = -axis_length * axis_offset;
    const end = axis_length + start;

    xLine.push([x_origin - end, start, start]);
    xLine.push([x_origin, start, start]);
    xLine['agent'] = agents[0];

    if (agents.length > 1) {
        yLine.push([x_origin, start, start]);
        yLine.push([x_origin, end, start]);
        yLine['agent'] = agents[1];
    } else { // Don't draw the axis if these agents are not included
        yLine.push([0, 0, 0]);
        yLine.push([0, 0, 0]);
    }

    if (agents.length > 2) {
        zLine.push([x_origin, start, start]);
        zLine.push([x_origin, start, end]);
        zLine['agent'] = agents[2];
    } else { // Don't draw the axis if these agents are not included
        zLine.push([0, 0, 0]);
        zLine.push([0, 0, 0]);
    }

    processData(cubes3D(cubesData), yScale3d([yLine]), xScale3d([xLine]), zScale3d([zLine]), 1000);


    function processData(cubeData, yScaleData, xScaleData, zScaleData, tt){

        /* --------- CUBES ---------*/

        var cubes = cubesGroup.selectAll('g.cube')
                        .data(cubeData, (d) => d.id);

        var ce = cubes
            .enter()
            .append('g')
            .attr('class', 'cube')
            .attr('fill', (d) => color(d.scenario))
            .attr('stroke', (d) => d3.color(color(d.scenario)).darker(2))
            .merge(cubes)
            .sort(cubes3D.sort);

        cubes.exit().remove();

        /* --------- FACES ---------*/

        var faces = cubes.merge(ce)
                        .selectAll('path.face')
                        .data((d) =>  d.faces, (d) => d.face);

        faces.enter()
            .append('path')
            .attr('class', 'face')
            .attr('fill-opacity', 0.95)
            .classed('_3d', true)
            .merge(faces)
            .transition().duration(tt)
            .attr('d', cubes3D.draw);

        faces.exit().remove();

        /* --------- TEXT ---------*/

        var texts = cubes.merge(ce).selectAll('text.text').data(function(d){
                var _t = d.faces.filter(function(d){
                    return d.face === 'top';
                });
                return [{height: d.height, scenario: d.scenario, centroid: _t[0].centroid}];
        });

        texts
            .enter()
            .append('text')
            .attr('class', 'text')
            .attr('dy', '-.7em')
            .attr('text-anchor', 'middle')
            .attr('font-family', 'sans-serif')
            .attr('font-weight', 'bolder')
            .attr('x', (d) => origin[0] + scale * d.centroid.x)
            .attr('y', (d) => origin[1] + scale * d.centroid.y)
            .classed('_3d', true)
            .merge(texts)
            .transition().duration(tt)
            .attr('fill', (d) => d3.color(color(d.scenario)).darker(1))
            .attr('stroke', 'none')
            .attr('x', (d) => origin[0] + scale * d.centroid.x)
            .attr('y', (d) => origin[1] + scale * d.centroid.y)
            .text((d) => d.scenario);

        texts.exit().remove();

        /* --------- SORT TEXT & FACES ---------*/

        ce.selectAll('._3d').sort(d3._3d().sort);

        /* Axis */

        /* ----------- y-Scale ----------- */
        /* todo: color these scales by the group colors*/

        var yScale = svg.selectAll('path.yScale')
                        .data(yScaleData);

        yScale
            .enter()
            .append('path')
            .attr('class', '_3d yScale')
            .merge(yScale)
            .attr('stroke', 'black')
            .attr('stroke-width', .5)
            .attr('d', yScale3d.draw);

        yScale.exit().remove();

        /* ----------- y-Scale Text ----------- */

        var yText = svg.selectAll('text.yText')
                       .data(yScaleData);

        yText
            .enter()
            .append('text')
            .attr('class', '_3d yText')
            .attr('dx', '.3em')
            .merge(yText)
            // .each((d) => d.centroid = {x: d.rotated.x, y: d.rotated.y, z: d.rotated.z})
            // ^ if labeling each point do this and provide data for yText as yScaleData[0]
            .attr('x', (d) => origin[0] + scale * d.centroid.x)
            .attr('y', (d) => origin[1] + scale * d.centroid.y)
            .text((d) => d.agent);

        yText.exit().remove();

        /* ----------- x-Scale ----------- */

        var xScale = svg.selectAll('path.xScale')
                        .data(xScaleData);

        xScale
            .enter()
            .append('path')
            .attr('class', '_3d xScale')
            .merge(xScale)
            .attr('stroke', 'black')
            .attr('stroke-width', .5)
            .attr('d', xScale3d.draw);

        xScale.exit().remove();

        /* ----------- x-Scale Text ----------- */

        var xText = svg.selectAll('text.xText')
                       .data(xScaleData);

        xText
            .enter()
            .append('text')
            .attr('class', '_3d xText')
            .attr('dx', '.3em')
            .merge(xText)
            .attr('x', (d) => origin[0] + scale * d.centroid.x)
            .attr('y', (d) => origin[1] + scale * d.centroid.y)
            .text((d) => d.agent);

        xText.exit().remove();

        /* ----------- x-Scale ----------- */

        var zScale = svg.selectAll('path.zScale')
                        .data(zScaleData);

        zScale
            .enter()
            .append('path')
            .attr('class', '_3d zScale')
            .merge(zScale)
            .attr('stroke', 'black')
            .attr('stroke-width', .5)
            .attr('d', zScale3d.draw);

        zScale.exit().remove();

        /* ----------- x-Scale Text ----------- */

        var zText = svg.selectAll('text.zText')
                       .data(zScaleData);

        zText
            .enter()
            .append('text')
            .attr('class', '_3d zText')
            .attr('dx', '.3em')
            .merge(zText)
            .attr('x', (d) => origin[0] + scale * d.centroid.x)
            .attr('y', (d) => origin[1] + scale * d.centroid.y)
            .text((d) => d.agent);

        zText.exit().remove();
    }

    function dragStart(event, d){
        mx = event.x;
        my = event.y;
    }

    function dragged(event, d){
        mouseX = mouseX || 0;
        mouseY = mouseY || 0;
        beta   = (event.x - mx + mouseX) * Math.PI / 230 ;
        alpha  = (event.y - my + mouseY) * Math.PI / 230  * (-1);
        var new_cubes = cubes3D.rotateY(beta + startAngle).rotateX(alpha - startAngle)(cubesData);
        var new_yScale = yScale3d.rotateY(beta + startAngle).rotateX(alpha - startAngle)([yLine]);
        var new_xScale = xScale3d.rotateY(beta + startAngle).rotateX(alpha - startAngle)([xLine]);
        var new_zScale = zScale3d.rotateY(beta + startAngle).rotateX(alpha - startAngle)([zLine]);
        processData(new_cubes, new_yScale, new_xScale, new_zScale, 0);
    }

    function dragEnd(event, d){
        mouseX = event.x - mx + mouseX;
        mouseY = event.y - my + mouseY;
    }

    function makeCube(x1, y1, z1, x2, y2, z2){
        return [
            {x: x1, y: y1, z: z1}, // FRONT TOP LEFT
            {x: x1, y: y2, z: z1}, // FRONT BOTTOM LEFT
            {x: x2, y: y2, z: z1}, // FRONT BOTTOM RIGHT
            {x: x2, y: y1, z: z1}, // FRONT TOP RIGHT
            {x: x1, y: y1, z: z2}, // BACK  TOP LEFT
            {x: x1, y: y2, z: z2}, // BACK  BOTTOM LEFT
            {x: x2, y: y2, z: z2}, // BACK  BOTTOM RIGHT
            {x: x2, y: y1, z: z2}, // BACK  TOP RIGHT
        ];
    }

    return svg;
}   
