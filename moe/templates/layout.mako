<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" xmlns:tal="http://xml.zope.org/namespaces/tal">
<head>
  <!-- meta info -->
  <title>MOE: The Metric Optimization Engine</title>
  <meta http-equiv="Content-Type" content="text/html;charset=UTF-8"/>
  <meta name="keywords" content="python web application" />
  <meta name="description" content="Metric Optimization Engine" />
  <!-- ico -->
  <link rel="shortcut icon" href="${request.static_url('moe:static/ico/favicon.ico')}" />
  <!-- bootstrap -->
  <link rel="stylesheet" href="${request.static_url('moe:static/css/bootstrap.css')}" />
  <link rel="script" href="${request.static_url('moe:static/js/bootstrap.js')}" />
  <script src="//code.jquery.com/jquery-1.10.2.js"></script>
  <!-- font-awesome -->
  <link rel="stylesheet" href="${request.static_url('moe:static/css/font-awesome.min.css')}" />
  <!-- d3 -->
  <link rel="script" href="//cdnjs.cloudflare.com/ajax/libs/d3/3.4.1/d3.js">
  <!-- background image -->
  <!-- <style type='text/css'>
      body {
          background-image: url("${request.static_url('moe:static/img/moe_standing.png')}");
          background-position: right bottom;
          background-repeat: no-repeat;
          }
  </style>
  -->
</head>
<body>
    <div class="navbar navbar-default navbar-fixed-top" role="navigation">
      <div class="container">
        <div class="navbar-header">
          <button type="button" class="navbar-toggle" data-toggle="collapse" data-target=".navbar-collapse">
            <span class="sr-only">Toggle navigation</span>
            <span class="icon-bar"></span>
            <span class="icon-bar"></span>
            <span class="icon-bar"></span>
          </button>
          <a class="navbar-brand" href="${request.route_url('home')}">MOE</a>
        </div>
        <div class="collapse navbar-collapse">
          <ul class="nav navbar-nav">
            <li ${'class="active"' if nav_active == 'home' else '' | n }><a href="${request.route_url('home')}">Home</a></li>
            <li ${'class="active"' if nav_active == 'about' else '' | n }><a href="${request.route_url('about')}">About</a></li>
            <li ${'class="active"' if nav_active == 'docs' else '' | n }><a href="${request.route_url('docs')}">Docs</a></li>
          </ul>
        </div><!--/.nav-collapse -->
      </div>
    </div>

  <div class="container">
      ${next.body()}
  </div>
</body>
</html>
