var optimalLearning = optimalLearning || {};

optimalLearning.errorAlert = function(jqXHR, textStatus, errorThrown) {
  if (jqXHR.responseText.indexOf('DOCTYPE') !== -1){
    alert("INTERNAL ERROR\nCheck console.");
  } else {
    alert(jqXHR.responseText);
  }
};
