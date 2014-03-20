function plot_graphs(gp_data, ei_raw_data, xvals, points_sampled) {

    var margin = {top: 20, right: 20, bottom: 30, left: 50},
        width = 900 - margin.left - margin.right,
        height = 300 - margin.top - margin.bottom;

    // ei-graph
    var svg_ei = d3.select(".ei-graph").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    // scales
    var x = d3.scale.linear()
        .domain([0, 1])
        .range([0, width]);

    var ei_data = []
    var i = 0;
    for (xpoint in xvals) {
        ei_data.push({
            'x': xvals[xpoint][0],
            'y': parseFloat(ei_raw_data["expected_improvement"][i]),
        });
        i += 1;
    }
    ei_data.sort(function(a,b){
        return a.x-b.x
    });

    var y = d3.scale.linear()
        .domain([
                0,
                d3.max(ei_data, function(d) { return d.y; }),
                ]
                )
        .range([height, 0]);

    // y axis
    var yAxis = d3.svg.axis()
        .scale(y)
        .orient("left");

    svg_ei.append("g")
      .attr("class", "y axis")
      .call(yAxis)
    .append("text")
      .attr("class", "label")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text("Expected Improvement");

    // x axis
    var xAxis = d3.svg.axis()
        .scale(x)
        .orient("bottom");

    svg_ei.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis)
    .append("text")
      .attr("class", "label")
      .attr("x", width)
      .attr("y", -6)
      .style("text-anchor", "end")
      .text("Dimension");

    var make_line = d3.svg.line()
        .x(function(d) { return x(d.x); })
        .y(function(d) { return y(d.y); })
        .interpolate("linear");

    var mu_line_graph = svg_ei.append("path")
        .attr("class", "line")
        .attr("d", make_line(ei_data))
        .attr("stroke", "black")
        .attr("stroke-width", 1)
        .attr("fill", "none");

    // gp-graph

    var svg = d3.select(".gp-graph").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var gp_var_upper_data = []
    for (xpoint in xvals) {
        gp_var_upper_data.push({
            'x': xvals[xpoint][0],
            'y0': parseFloat(gp_data['mean'][xpoint]),
            'y1': parseFloat(gp_data['mean'][xpoint]) + parseFloat(gp_data['var'][xpoint]),
            'bound': parseFloat(gp_data['mean'][xpoint]) + parseFloat(gp_data['var'][xpoint]),
        });
    }
    gp_var_upper_data.sort(function(a,b){
        return a.x-b.x
    });

    var gp_var_lower_data = []
    for (xpoint in xvals) {
        gp_var_lower_data.push({
            'x': xvals[xpoint][0],
            'y0': parseFloat(gp_data['mean'][xpoint]) - parseFloat(gp_data['var'][xpoint]),
            'y1': parseFloat(gp_data['mean'][xpoint]),
            'bound': parseFloat(gp_data['mean'][xpoint]) - parseFloat(gp_data['var'][xpoint]),
        });
    }
    gp_var_lower_data.sort(function(a,b){
        return a.x-b.x
    });

    var gp_mu_data = []
    for (xpoint in xvals) {
        gp_mu_data.push({
            'x': xvals[xpoint][0],
            'y': parseFloat(gp_data['mean'][xpoint]),
        });
    }
    gp_mu_data.sort(function(a,b){
        return a.x-b.x
    });

    // scales
    var x = d3.scale.linear()
        .domain([0, 1])
        .range([0, width]);

    var y = d3.scale.linear()
        .domain([
                d3.min(gp_var_lower_data, function(d) { return d.y0 - 0.05; }),
                d3.max(gp_var_upper_data, function(d) { return d.y1 + 0.05; }),
                ]
                )
        .range([height, 0]);

    // variance areas
    var area = d3.svg.area()
        .x(function(d) { return x(d.x); })
        .y0(function(d) { return y(d.y0); })
        .y1(function(d) { return y(d.y1); });

    function make_gp_var_plot(data){
        svg.append("path")
            .attr("d", area(data))
            .style("fill", "#CBDDB9");
    }
    make_gp_var_plot(gp_var_upper_data);
    make_gp_var_plot(gp_var_lower_data);

    // mu line
    var mu_line_graph = svg.append("path")
        .attr("class", "line")
        .style("stroke-dasharray", ("5, 3"))
        .attr("d", make_line(gp_mu_data))
        .attr("stroke", "black")
        .attr("stroke-width", 1)
        .attr("fill", "none");

    // points sampled
    svg.selectAll("scatter-dots")
      .data(points_sampled)  // using the values in the ydata array
      .enter().append("svg:circle")  // create a new circle for each value
          .attr("cy", function (d) { return y(parseFloat(d['value'])); } ) // translate y value to a pixel
          .attr("cx", function (d) { return x(parseFloat(d['point'][0])); } ) // translate x value
          .attr("r", 2) // radius of circle
          .attr("stroke", "black")
          .attr("fill", "black");

    // error bars
    var error_bar_width = 0.005;
    var error_bar_color = "rgb(0,0,0)"
    data = points_sampled;
    for (var i in points_sampled) {
        // vertical line
        x_val = parseFloat(data[i]['point'][0]);
        y_val = parseFloat(data[i]['value']);
        v_val = parseFloat(data[i]['value_var']);
        svg.append("svg:line")
            .attr("x1", x(x_val))
            .attr("y1", y(y_val + v_val/2.0))
            .attr("x2", x(x_val))
            .attr("y2", y(y_val - v_val/2.0))
            .style("stroke", error_bar_color);
        // upper cross line
        svg.append("svg:line")
            .attr("x1", x(x_val + error_bar_width))
            .attr("y1", y(y_val + v_val/2.0))
            .attr("x2", x(x_val - error_bar_width))
            .attr("y2", y(y_val + v_val/2.0))
            .style("stroke", error_bar_color);
        // lower cross line
        svg.append("svg:line")
            .attr("x1", x(x_val + error_bar_width))
            .attr("y1", y(y_val - v_val/2.0))
            .attr("x2", x(x_val - error_bar_width))
            .attr("y2", y(y_val - v_val/2.0))
            .style("stroke", error_bar_color);
    }

    // y axis
    var yAxis = d3.svg.axis()
        .scale(y)
        .orient("left");

    svg.append("g")
      .attr("class", "y axis")
      .call(yAxis)
    .append("text")
      .attr("class", "label")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text("Objective Function");

    // x axis
    var xAxis = d3.svg.axis()
        .scale(x)
        .orient("bottom");

    svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis)
    .append("text")
      .attr("class", "label")
      .attr("x", width)
      .attr("y", -6)
      .style("text-anchor", "end")
      .text("Dimension");
}
