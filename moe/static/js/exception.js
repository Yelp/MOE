var optimalLearning = optimalLearning || {};

optimalLearning.errorAlert = function(jqXHR, textStatus, errorThrown) {
  if (jqXHR.responseText.indexOf('DOCTYPE') !== -1){
    alert("INTERNAL 500 ERROR\nCheck console.");
  } else {
    alert("500 ERROR\n" + jqXHR.responseText);
  }
};
