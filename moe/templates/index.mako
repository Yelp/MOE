<%inherit file="moe:templates/layout.mako"/>

<div class="background-image">
  <h1>MOE: Metric Optimization Engine</h1>
  <h4>A global, black box optimization framework</h4>
  <br>

    <div class="span12">
      <div class="row">
        <div class="col-md-5">
          <div class="panel panel-info">
            <div class="panel-heading"><b>Documentation</b></div>
              <ul class="list-group">
                <a href="http://yelp.github.io/MOE/" class="list-group-item">Full Online Documentation</a>
                <a href="http://yelp.github.io/MOE/moe.views.rest.html" class="list-group-item">REST Endpoint Documentation</a>
                <a href="http://github.com/Yelp/MOE/" class="list-group-item">Github repo</a>
                <a href="${request.route_url( 'gp_plot' )}" class="list-group-item">Interactive Demo</a>
              </ul>
          </div>
        </div>

        <div class="col-md-6">
          <div class="panel panel-success">
            <div class="panel-heading"><b>Gaussian Process (GP) Pretty Endpoints</b></div>
            <div class="panel-body">
              <p>Interactive forms for trying and testing the various MOE REST endpoints.</p>
            </div>
              <ul class="list-group">
                <li class="list-group-item"><b>Gaussian Process (GP) Posterior Mean and [Co]Variance Endpoints</b></li>
                <a href="${request.route_url( 'gp_mean_pretty' )}" class="list-group-item">GP mean</a>
                <a href="${request.route_url( 'gp_var_pretty' )}" class="list-group-item">GP covariance</a>
                <a href="${request.route_url( 'gp_var_diag_pretty' )}" class="list-group-item">GP variance</a>
                <a href="${request.route_url( 'gp_mean_var_pretty' )}" class="list-group-item">GP mean and covariance</a>
                <a href="${request.route_url( 'gp_mean_var_diag_pretty' )}" class="list-group-item">GP mean and variance</a>

                <li class="list-group-item"><b>Optimal Next Points to Sample Endpoints</b></li>
                <a href="${request.route_url( 'gp_next_points_epi_pretty' )}" class="list-group-item">GP next points EPI</a>
                <a href="${request.route_url( 'gp_next_points_kriging_pretty' )}" class="list-group-item">GP next points kriging</a>
                <a href="${request.route_url( 'gp_next_points_constant_liar_pretty' )}" class="list-group-item">GP next points constant liar</a>

                <li class="list-group-item"><b>Expected Improvement Endpoints</b></li>
                <a href="${request.route_url( 'gp_ei_pretty' )}" class="list-group-item">GP EI</a>

                <li class="list-group-item"><b>Gaussian Process (GP) Hyperparameter Optimization Endpoints</b></li>
                <a href="${request.route_url( 'gp_hyper_opt_pretty' )}" class="list-group-item">GP hyperparameter optimization</a>
                
              </ul>
            </div>

          <div class="panel panel-success">
            <div class="panel-heading"><b>Bandit Pretty Endpoints</b></div>
            <div class="panel-body">
              <p>Interactive forms for trying and testing the various MOE REST endpoints.</p>
            </div>
              <ul class="list-group">
                <li class="list-group-item"><b>Bandit Endpoints</b></li>
                <a href="${request.route_url( 'bandit_epsilon_pretty' )}" class="list-group-item">Bandit epsilon</a>
                <a href="${request.route_url( 'bandit_ucb_pretty' )}" class="list-group-item">Bandit UCB</a>
                
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
</div>
