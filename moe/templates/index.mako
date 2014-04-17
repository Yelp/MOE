<%inherit file="moe:templates/layout.mako"/>

<div class="background-image">
    <h2>MOE: Metric Optimization Engine</h2>
    <p>A global, black box optimization framework</p>
    <br>

    <h4>Documentation</h4>
    <ul>
        <li><a href="http://sc932.github.io/MOE/">Full Documentation</a></li>
        <li><a href="http://github.com/sc932/MOE/">Github repo</a></li>
        <li><a href="${request.route_url( 'gp_plot' )}">Demo</a></li>
    </ul>

    <h4>Pretty Endpoints</h4>
    <ul>
        <li><a href="${request.route_url( 'gp_ei_pretty' )}">GP EI</a></li>
        <li><a href="${request.route_url( 'gp_mean_var_pretty' )}">GP mean and var</a></li>
        <li><a href="${request.route_url( 'gp_next_points_epi_pretty' )}">GP next points EPI</a></li>
        <li><a href="${request.route_url( 'gp_next_points_kriging_pretty' )}">GP next points kriging</a></li>
        <li><a href="${request.route_url( 'gp_next_points_constant_liar_pretty' )}">GP next points constant liar</a></li>
    </ul>
</div>
