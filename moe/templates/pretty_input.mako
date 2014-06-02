<%inherit file="moe:templates/layout.mako"/>

<div class="span12">
    <h4>Pretty input for <strong><code>${ endpoint }</code></strong> endpoint</h4>
    <p><a href="http://sc932.github.io/MOE/moe.views.rest.html#module-moe.views.rest.${ endpoint }">Documentation</a>
    <p><textarea class="form-control mono-form" id="json-body" rows="10">${ default_text }</textarea></p>
    <p><button id="submit" type="button" class="btn btn-success">Submit</button></p>
    <p><textarea class="form-control mono-form" id="json-out" rows="10" placeholder="Output JSON"></textarea></p>
    <div id="loading-screen"></div>
</div>

<script>
$("#submit").click(function() {
    $("#loading-screen").html('<h1>Processing...</h1><div class="progress progress-striped active"><div class="progress-bar"  role="progressbar" aria-valuenow="100" aria-valuemin="0" aria-valuemax="100" style="width: 100%"><span class="sr-only">100% Complete</span></div></div>');
    var jqxhr = $.post(
        "${request.route_url( endpoint )}",
        $("#json-body").val(),
        function( data ) {
            $("#json-out").val( JSON.stringify(data) );
        }
    );
    jqxhr.fail(function(jqXHR, textStatus, errorThrown) {
        if (jqXHR.responseText.indexOf('DOCTYPE') !== -1){
            alert("INTERNAL 500 ERROR\nCheck console.");
        }else{
            alert("500 ERROR\n" + jqXHR.responseText);
        }
    });
    jqxhr.done(function() {
        $("#loading-screen").html('');
    });
});
</script>
    
