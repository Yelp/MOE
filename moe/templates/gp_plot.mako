<%inherit file="moe:templates/layout.mako"/>
<div class="span12">

    <div class="row">
        <div class="col-md-8">
            <div id="graph-area">
                <div class="row">
                    <div class="col-md-6">
                        <h3>
                            Gaussian Process (GP)
                            <span class="glyphicon glyphicon-question-sign tooltip-rdy small" data-original-title="The Gaussian Process (GPs) posterior mean and variance given the historical data and parameters (on right). The dashed line is the posterior mean, the faded area is the variance, for each point in [0,1]." data-placement="bottom"><span>
                        </h3>
                    </div>
                    <div class="col-md-6 middle-text">
                        Endpoint(s):
                        <a href="http://yelp.github.io/MOE/moe.views.rest.html#moe.views.rest.gp_mean_var.GpMeanVarDiagView"><strong><code>gp_mean_var_diag</code></strong></a>
                        <span class="glyphicon glyphicon-question-sign tooltip-rdy small" data-original-title="The endpoint(s) (and docs) used to generate the below graph." data-placement="top"><span>
                    </div>
                </div>
                <div class="gp-graph"></div>
                <div class="row">
                    <div class="col-md-6">
                        <h3>
                            Expected Improvement (EI)
                            <span class="glyphicon glyphicon-question-sign tooltip-rdy small" data-original-title="A plot of Expected Improvement (EI) for each potential next point in [0,1]. The red line corresponds to the point of highest EI within the domain. This is the point that MOE suggest we sample next to optimize EI." data-placement="top"><span>
                        </h3>
                    </div>
                    <div class="col-md-6 middle-text">
                        Endpoint(s):
                        <a href="http://yelp.github.io/MOE/moe.views.rest.html#module-moe.views.rest.gp_ei"><strong><code>gp_ei</code></strong></a> and <a href="http://yelp.github.io/MOE/moe.views.rest.html#module-moe.views.rest.gp_next_points_epi"><strong><code>gp_next_points_epi</code></strong></a>
                        <span class="glyphicon glyphicon-question-sign tooltip-rdy small" data-original-title="The endpoint(s) (and docs) used to generate the below graph." data-placement="top"><span>
                    </div>
                </div>
                <div class="ei-graph"></div>
            </div>
        </div>
        <div class="col-md-4">
            <center>
                <h4>
                    GP Parameters
                    <span class="glyphicon glyphicon-question-sign tooltip-rdy small" data-original-title="The hyperparameters for the Gaussian Process (GP). These parameters determine model fit and can be different depending on the covariance kernel." data-placement="bottom"><span>
                </h4>
            </center>
            <form class="form-horizontal" role="form">
              <div class="form-group">
                <label for="hyperparameters-alpha" class="col-sm-6 control-label">
                    Signal Variance
                    <span class="glyphicon glyphicon-question-sign tooltip-rdy small" data-original-title="Signal Variance is a measure of the underlying uncertainty of the Gaussian Process (GP) Prior." data-placement="bottom"><span>
                </label>
                <div class="col-sm-6">
                  <input class="form-control" id="hyperparameters-alpha" value="${ default_gaussian_process_parameters.signal_variance }">
                </div>
              </div>
              <div class="form-group">
                <label for="hyperparameters-length" class="col-sm-6 control-label">
                    Length Scale
                    <span class="glyphicon glyphicon-question-sign tooltip-rdy small" data-original-title="Length Scale determines how closely correlated two sampled points are. Higher length scales will result in underfitting, lower length scales will result in overfitting." data-placement="bottom"><span>
                </label>
                <div class="col-sm-6">
                  <input class="form-control" id="hyperparameters-length" value="${ default_gaussian_process_parameters.length_scale[0] }">
                </div>
              </div>
            </form>
            <center>
                <h4>
                    EI SGD Parameters
                    <span class="glyphicon glyphicon-question-sign tooltip-rdy small" data-original-title="The Expected Improvement (EI) Stochastic Gradient Descent (SGD) Parameters. These parameters change how the underlying EI optimization function functions." data-placement="bottom"><span>
                </h4>
            </center>
            <form class="form-horizontal" role="form">
              <div class="form-group">
                <label for="opt-num-multistarts" class="col-sm-6 control-label">
                    Multistarts
                    <span class="glyphicon glyphicon-question-sign tooltip-rdy small" data-original-title="The number of multistarts that the SGD algorithm will use. Higher numbers will potentialily have the algorithm explore more local EI optima at extra computational cost." data-placement="bottom"><span>
                </label>
                <div class="col-sm-6">
                  <input class="form-control" id="opt-num-multistarts" value="${ default_num_multistarts }">
                </div>
              </div>
              <div class="form-group">
                <label for="opt-gd-iterations" class="col-sm-6 control-label">
                    GD Iterations
                    <span class="glyphicon glyphicon-question-sign tooltip-rdy small" data-original-title="Controls how many Gradient Descent (GD) steps the SGD algorithm will take for each multistart. Higher values will resolve individual local optima more." data-placement="bottom"><span>
                </label>
                <div class="col-sm-6">
                  <input class="form-control" id="opt-gd-iterations" value="${ default_ei_optimizer_parameters.max_num_steps }">
                </div>
              </div>
            </form>
            <center>
                <button id="submit" type="button" class="btn btn-success">Apply Parameter Updates</button>
            </center>
            <hr>
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
            <center>
                <button id="add-point" type="button" class="btn btn-primary">Add Point</button>
                <span class="glyphicon glyphicon-question-sign tooltip-rdy small" data-original-title="Adds the above point to the historical points sampled. By default it will suggest the point of highest EI (red line on left) and sample a value from the GP." data-placement="bottom"><span>
            </center>
            <div id="loading-screen"></div>
            <hr>
            <center>
                <h4>
                    Points Sampled
                    <span class="glyphicon glyphicon-question-sign tooltip-rdy small" data-original-title="The historical points that have been sampled. In the form f(point) = value &plusmn; variance." data-placement="bottom"><span>
                </h4>
            </center>
            <div id="points-sampled">
                <ul>

                </ul>
            </div>
        </div>
    </div>
</div>

<script language="javascript" type="text/javascript" src="${request.static_url('moe:static/js/gp_plot.js')}"></script>
<script language="javascript" type="text/javascript" src="${request.static_url('moe:static/js/exception.js')}"></script>
<script>
var optimalLearning = optimalLearning || {};

/*
 *  normRand: returns normally distributed random numbers
 */
optimalLearning.normRand = function() {
  var x1, x2, rad;

  do {
      x1 = 2 * Math.random() - 1;
      x2 = 2 * Math.random() - 1;
      rad = x1 * x1 + x2 * x2;
  } while(rad >= 1 || rad == 0);

  var c = Math.sqrt(-2 * Math.log(rad) / rad);

  return x1 * c;
};

optimalLearning.pointsSampled = [];

optimalLearning.updateGraphs = function() {
    var xvals = [];
    for (i = 0; i <= 200; i++) {
        xvals.push( [i / 200.0] );
    }

    var gp_post_data = {
        'points_to_evaluate': xvals,
        'gp_historical_info': {
            'points_sampled': optimalLearning.pointsSampled,
            },
        'domain_info': {
            'dim': 1,
            },
        'covariance_info': {
            'covariance_type': 'square_exponential',
            'hyperparameters': [
                $.parseJSON($('#hyperparameters-alpha').val()),
                $.parseJSON($('#hyperparameters-length').val())
                ],
            },
        }
    var gp_optimize_post_data = {
        'gp_historical_info': {
            'points_sampled': optimalLearning.pointsSampled,
            },
        'domain_info': {
            'dim': 1,
            'domain_bounds': [
                {'min': 0.0, 'max': 1.0},
                ],
            },
        'covariance_info': {
            'covariance_type': 'square_exponential',
            'hyperparameters': [
                $.parseJSON($('#hyperparameters-alpha').val()),
                $.parseJSON($('#hyperparameters-length').val())
                ],
            },
        'optimizer_info':{
            'optimizer_type': 'gradient_descent_optimizer',
            'num_multistarts': $.parseJSON($('#opt-num-multistarts').val()),
            'num_random_samples': 400,
            'optimizer_parameters': {
                'max_num_steps': $.parseJSON($('#opt-gd-iterations').val()),
                },
            },
        }

    var gp_data, ei_raw_data, next_points_raw_data, single_point_gp_data;
    var jqxhr1 = $.post(
        "${request.route_url('gp_mean_var_diag')}",
        JSON.stringify(gp_post_data),
        function( data ) {
            gp_data = data;
        }
    );
    jqxhr1.fail(optimalLearning.errorAlert);

    var jqxhr2 = $.post(
        "${request.route_url('gp_ei')}",
        JSON.stringify(gp_post_data),
        function( data ) {
            ei_raw_data = data;
        }
    );
    jqxhr2.fail(optimalLearning.errorAlert);

    gp_optimize_post_data['num_to_sample'] = 1;
    var jqxhr3 = $.post(
        "${request.route_url('gp_next_points_epi')}",
        JSON.stringify(gp_optimize_post_data),
        function( data ) {
            next_points_raw_data = data;
        }
    );
    jqxhr3.fail(optimalLearning.errorAlert);

    $("#loading-screen").html('<h1>Processing...</h1><div class="progress progress-striped active"><div class="progress-bar"  role="progressbar" aria-valuenow="100" aria-valuemin="0" aria-valuemax="100" style="width: 100%"><span class="sr-only">100% Complete</span></div></div>');
    
    jqxhr3.done(function() {
        $("#loading-screen").html('');
        var x_value = $.parseJSON(next_points_raw_data['points_to_sample'][0][0]);
        $("#x1").val(x_value.toFixed(4));
        
        jqxhr1.done(function() {
            jqxhr2.done(function() {
                $(".gp-graph").html("");
                $(".ei-graph").html("");
                optimalLearning.plotGraphs(gp_data, ei_raw_data, xvals, optimalLearning.pointsSampled, next_points_raw_data['points_to_sample'][0][0]);
            });
        });

        var single_point_gp_data;
        gp_post_data['points_to_evaluate'] = next_points_raw_data['points_to_sample'];
        var jqxhr4 = $.post(
            "${request.route_url('gp_mean_var_diag')}",
            JSON.stringify(gp_post_data),
            function( data ) {
                single_point_gp_data = data;
            }
        );
        jqxhr4.fail(optimalLearning.errorAlert);

        jqxhr4.done(function() {
            var y_value = $.parseJSON(single_point_gp_data['mean'][0]) + $.parseJSON(single_point_gp_data['var'][0]) * optimalLearning.normRand();
            $("#y1").val(y_value.toFixed(4));
            $("#v1").val('0.1000');
        });
    });
};

optimalLearning.updatePointsSampled = function() {
    $('#points-sampled ul').html('');
    for (i in optimalLearning.pointsSampled){
        point = optimalLearning.pointsSampled[i];
        $('#points-sampled ul').append('<li id=' + i + '>f(' + point.point[0].toFixed(4) + ') = ' + point.value.toFixed(4) + ' &plusmn ' + point.value_var.toFixed(4) + ' <a class="itemDelete"><button type="button" class="btn btn-danger btn-xs">remove</button></a></li>');
    }
    optimalLearning.updateGraphs();

    $('.itemDelete').click(function() {
        var idx = $.parseJSON($(this).closest('li')[0].id);
        optimalLearning.pointsSampled.splice(idx, 1);
        console.log(optimalLearning.pointsSampled);
        optimalLearning.updatePointsSampled();
    });
};

optimalLearning.addPointClickAction = function() {
  optimalLearning.pointsSampled.push(
      {
          'point': [
              $.parseJSON($("#x1").val())
              ],
          'value': $.parseJSON($("#y1").val()),
          'value_var': $.parseJSON($("#v1").val()),
      }
  );
  console.log(optimalLearning.pointsSampled);
  optimalLearning.updatePointsSampled();
};

$("#add-point").click(optimalLearning.addPointClickAction);

optimalLearning.updatePointsSampled();

console.log(optimalLearning.pointsSampled);

$("#submit").click(function() {
    optimalLearning.updateGraphs();
});

$(document).ready(function() {
    $(".tooltip-rdy").tooltip();
});
</script>
