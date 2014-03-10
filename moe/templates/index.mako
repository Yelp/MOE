<%inherit file="moe:templates/layout.mako"/>

Home

<h4>Pretty Endpoints</h4>
<ul>
    <li><a href="${request.route_url( 'gp_ei_pretty' )}">GP EI</a></li>
    <li><a href="${request.route_url( 'gp_mean_var_pretty' )}">GP mean and var</a></li>
    <li><a href="${request.route_url( 'gp_next_points_epi_pretty' )}">GP next points EPI</a></li>
</ul>
