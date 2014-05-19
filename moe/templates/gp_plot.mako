<%inherit file="moe:templates/layout.mako"/>
<div class="span12">

    <div class="row">
        <div class="col-md-8">
            <div id="graph-area">
                <h3>Gaussian Process</h3>
                <div class="gp-graph"></div>
                <h3>Expected Improvement</h3>
                <div class="ei-graph"></div>
            </div>
        </div>
        <div class="col-md-4">
            <center><h4>GP Parameters</h4></center>
            <form class="form-horizontal" role="form">
              <div class="form-group">
                <label for="hyperparameters-alpha" class="col-sm-6 control-label">Signal Variance</label>
                <div class="col-sm-6">
                  <input class="form-control" id="hyperparameters-alpha" value="${ default_gaussian_process_parameters.signal_variance }">
                </div>
              </div>
              <div class="form-group">
                <label for="hyperparameters-length" class="col-sm-6 control-label">Length Scale</label>
                <div class="col-sm-6">
                  <input class="form-control" id="hyperparameters-length" value="${ default_gaussian_process_parameters.length_scale[0] }">
                </div>
              </div>
            </form>
            <center><h4>EI SGD Parameters</h4></center>
            <form class="form-horizontal" role="form">
              <div class="form-group">
                <label for="opt-num-multistarts" class="col-sm-6 control-label">Multistarts</label>
                <div class="col-sm-6">
                  <input class="form-control" id="opt-num-multistarts" value="${ default_ei_optimization_parameters.num_multistarts }">
                </div>
              </div>
              <div class="form-group">
                <label for="opt-gd-iterations" class="col-sm-6 control-label">GD Iterations</label>
                <div class="col-sm-6">
                  <input class="form-control" id="opt-gd-iterations" value="${ default_ei_optimization_parameters.gd_iterations }">
                </div>
              </div>
              <div class="form-group">
                <label for="opt-pre-mult" class="col-sm-6 control-label">Pre-mult</label>
                <div class="col-sm-6">
                  <input class="form-control" id="opt-pre-mult" value="${ default_ei_optimization_parameters.pre_mult }">
                </div>
              </div>
              <div class="form-group">
                <label for="opt-gamma" class="col-sm-6 control-label">Gamma</label>
                <div class="col-sm-6">
                  <input class="form-control" id="opt-gamma" value="${ default_ei_optimization_parameters.gamma }">
                </div>
              </div>
            </form>
            <form class="form-horizontal" role="form">
              <div class="form-group">
                <label for="make-graphs" class="col-sm-6 control-label">Update Graphs</label>
                <div class="col-sm-6">
                    <button id="submit" type="button" class="btn btn-success">Submit</button>
                </div>
              </div>
            </form>
            <hr>
            <center><h4>Points Sampled</h4></center>
            <div id="points-sampled">
                <ul>

                </ul>
            </div>
            <div id="point-inputs">
                <form class="form-inline" role="form">
                    f(
                    <div class="form-group">
                        <input class="form-control" id="x1" placeholder="x">
                    </div>
                    ) = 
                    <div class="form-group">
                        <input class="form-control" id="y1" placeholder="y">
                    </div>
                     &plusmn; 
                    <div class="form-group">
                        <input class="form-control" id="v1" placeholder="&sigma;">
                    </div>
                </form>
            </div>
            <br>
            <p>
                <button id="add-point" type="button" class="btn btn-primary">Add Point</button>
                <br>(default: point of highest EI sampled from process)
            </p>
            <div id="loading-screen"></div>
        </div>
    </div>
</div>

<script>
var points_sampled = [];

function update_points_sampled(){
    $('#points-sampled ul').html('');
    for (i in points_sampled){
        point = points_sampled[i];
        $('#points-sampled ul').append('<li id=' + i + '>f(' + point.point[0].toFixed(4) + ') = ' + point.value.toFixed(4) + ' &plusmn ' + point.value_var.toFixed(4) + ' <a class="itemDelete"><button type="button" class="btn btn-danger btn-xs">remove</button></a></li>');
    }
    update_graphs();

    $('.itemDelete').click(function() {
        var idx = $.parseJSON($(this).closest('li')[0].id);
        points_sampled.splice(idx, 1);
        console.log(points_sampled);
        update_points_sampled();
    });
}

$("#add-point").click(function() {
    points_sampled.push(
        {
            'point': [
                $.parseJSON($("#x1").val())
                ],
            'value': $.parseJSON($("#y1").val()),
            'value_var': $.parseJSON($("#v1").val()),
        }
    );
    console.log(points_sampled);
    update_points_sampled();
});

update_points_sampled();

console.log(points_sampled);

$("#submit").click(function() {
    update_graphs();
});

function update_graphs(){
    var xvals = [];
    for (i = 0; i <= 200; i++) {
        xvals.push( [i / 200.0] );
    }

    var post_data = {
        'points_to_sample': xvals,
        'points_to_evaluate': xvals,
        'gp_info': {
            'points_sampled': points_sampled,
            'domain': [[0, 1]],
            'length_scale': [$.parseJSON($('#hyperparameters-length').val())],
            'signal_variance': $.parseJSON($('#hyperparameters-alpha').val()),
            },
        'ei_optimization_parameters': {
            'num_multistarts': $.parseJSON($('#opt-num-multistarts').val()),
            'gd_iterations': $.parseJSON($('#opt-gd-iterations').val()),
            },
        }

    console.log(post_data);

    var gp_data, ei_raw_data, next_points_raw_data, single_point_gp_data;
    var jqxhr1 = $.post(
        "${request.route_url('gp_mean_var')}",
        JSON.stringify(post_data),
        function( data ) {
            gp_data = data;
        }
    );
    jqxhr1.fail(function() {
        alert("500 error 1");
    });

    var jqxhr2 = $.post(
        "${request.route_url('gp_ei')}",
        JSON.stringify(post_data),
        function( data ) {
            ei_raw_data = data;
        }
    );
    jqxhr2.fail(function() {
        alert("500 error 2");
    });

    post_data['num_to_sample'] = 1;
    var jqxhr3 = $.post(
        "${request.route_url('gp_next_points_epi')}",
        JSON.stringify(post_data),
        function( data ) {
            next_points_raw_data = data;
        }
    );
    jqxhr3.fail(function() {
        alert("500 error 3");
    });

    $("#loading-screen").html('<h1>Processing...</h1><div class="progress progress-striped active"><div class="progress-bar"  role="progressbar" aria-valuenow="100" aria-valuemin="0" aria-valuemax="100" style="width: 100%"><span class="sr-only">100% Complete</span></div></div>');
    
    jqxhr3.done(function() {
        $("#loading-screen").html('');
        var x_value = $.parseJSON(next_points_raw_data['points_to_sample'][0][0]);
        $("#x1").val(x_value.toFixed(4));
        
        jqxhr1.done(function() {
            jqxhr2.done(function() {
                $(".gp-graph").html("");
                $(".ei-graph").html("");
                plot_graphs(gp_data, ei_raw_data, xvals, points_sampled, next_points_raw_data['points_to_sample'][0][0]);
            });
        });

        var single_point_gp_data;
        post_data['points_to_sample'] = next_points_raw_data['points_to_sample'];
        var jqxhr4 = $.post(
            "${request.route_url('gp_mean_var')}",
            JSON.stringify(post_data),
            function( data ) {
                single_point_gp_data = data;
            }
        );
        jqxhr4.fail(function() {
            alert("500 error 4");
        });

        jqxhr4.done(function() {
            var y_value = $.parseJSON(single_point_gp_data['mean'][0]) + $.parseJSON(single_point_gp_data['var'][0][0]) * normRand();
            $("#y1").val(y_value.toFixed(4));
            $("#v1").val('0.1000');
        });
    });
}

/*
 *  normRand: returns normally distributed random numbers
 */
function normRand() {
    var x1, x2, rad;
 
    do {
        x1 = 2 * Math.random() - 1;
        x2 = 2 * Math.random() - 1;
        rad = x1 * x1 + x2 * x2;
    } while(rad >= 1 || rad == 0);
 
    var c = Math.sqrt(-2 * Math.log(rad) / rad);
 
    return x1 * c;
};
</script>
<script language="javascript" type="text/javascript" src="${request.static_url('moe:static/js/gp_plot.js')}"></script>
