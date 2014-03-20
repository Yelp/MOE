<%inherit file="moe:templates/layout.mako"/>

<p>Gaussian Process</p>
<div class="gp-graph"></div>
<p>Expected Improvement</p>
<div class="ei-graph"></div>
<div class="span12">
    <center><h4>Parameters</h4></center>
    <br>
    <div class="row">
        <div class="col-md-6">
            <form class="form-horizontal" role="form">
              <div class="form-group">
                <label for="hyperparameters" class="col-sm-6 control-label">Hyperparameters</label>
                <div class="col-sm-6">
                  <input class="form-control" id="hyperparameters" value="[1.0, 1.0]">
                </div>
              </div>
              <div class="form-group">
                <label for="MC-iterations" class="col-sm-6 control-label">MC iterations</label>
                <div class="col-sm-6">
                  <input class="form-control" id="mc_iterations" value="1000">
                </div>
              </div>
            </form>
        </div>
        <div class="col-md-6">
            <p><textarea class="form-control" id="json-body" rows="1">${ default_text }</textarea></p>
            <h4>Points Sampled</h4>
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
            <p><button id="add-point" type="button" class="btn btn-default">Add Point</button></p>
        </div>
    </div>
    <p><button id="submit" type="button" class="btn btn-success">Submit</button></p>
</div>

<script>
$("#submit").click(function() {
    var post_data = $.parseJSON($("#json-body").val());
    var xvals = [];
    for (i = 0; i < 200; i++) {
        xvals.push( [i / 200.0] );
    }
    post_data['points_to_sample'] = xvals;
    post_data['points_to_evaluate'] = xvals;
    console.log(JSON.stringify(post_data));
    var gp_data, ei_raw_data;
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

    jqxhr1.done(function() {
        jqxhr2.done(function() {
            $(".gp-graph").html("");
            $(".ei-graph").html("");
            points_sampled = $.parseJSON($("#json-body").val())['gp_info']['points_sampled'];
            plot_graphs(gp_data, ei_raw_data, xvals, points_sampled);
        });
    });


});
</script>
<script language="javascript" type="text/javascript" src="${request.static_url('moe:static/js/gp_plot.js')}"></script>
