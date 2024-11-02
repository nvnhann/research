var checkTitle = require("./titleChecker").checkTitle;

var handleSubmit = function() {
  // Check what text was put into the form field
  var formText = document.getElementById("title").value;

  if (!checkTitle(formText)) return;

  console.log("::: Form Submitted :::" + ":" + formText);

  // Create a new XMLHttpRequest object
  var xhr = new XMLHttpRequest();

  // Set the request method and URL
  xhr.open("POST", "http://localhost:8080/api/getall");
  xhr.setRequestHeader("Content-Type", "application/json");

  // Set up the callback function for when the request completes
  xhr.onload = function () {
    if (xhr.status === 200) {
      // Request was successful
      var response = JSON.parse(xhr.responseText);
      console.log(response);

      var str = "";
      response.data.map(function(rs) {
        str += "<div class=\"item\">"
          + "<p><b>Title:</b> " + rs.title + "</p>"
          + "<p><b>Author:</b> " + rs.author.name + "</p>"
          + "<p><b>Content:</b> " + rs.body + "</p>"
          + "</div>";
      });

      if (response.data.length === 0) str = "<p>Not found</p>";

      document.getElementById("results").innerHTML = str;
    } else {
      // Request failed
      console.error("Request failed. Status code: " + xhr.status);
    }
  };

  // Set up the request payload, if needed
  var data = JSON.stringify({ title: formText });

  // Send the request
  xhr.send(data);
};

module.exports = {
  handleSubmit: handleSubmit
};
