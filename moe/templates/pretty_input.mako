<%inherit file="moe:templates/layout.mako"/>

<div class="span12">
    <h4>Pretty input for <strong><code>${ endpoint }</code></strong> endpoint</h4>
    <p><a href="http://sc932.github.io/MOE/moe.views.html#module-moe.views.${ endpoint }">Documentation</a>
    <p><textarea class="form-control" id="json-body" rows="10">${ default_text }</textarea></p>
    <p><button id="submit" type="button" class="btn btn-success">Submit</button></p>
    <p><textarea class="form-control" id="json-out" rows="10" placeholder="Output JSON"></textarea></p>
</div>

<script>
$("#submit").click(function() {
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
});
</script>
    
