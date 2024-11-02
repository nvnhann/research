// handleSubmit.test.js
var handleSubmit = require("../src/client/js/formHandler").handleSubmit;

describe("handleSubmit", () => {
  beforeEach(() => {
    // Create a mock for the form field
    document.body.innerHTML = `
      <input id="title" type="text" />
      <div id="results"></div>
    `;
  });

  test("should not submit if formText is empty", () => {
    // Set the formText value to an empty string
    document.getElementById("title").value = "";

    // Mock the console.log function
    console.log = jest.fn();

    // Call the handleSubmit function
    handleSubmit();

    // Verify that the console.log function was called with the expected message
    expect(console.log).toHaveBeenCalledWith("::: Running checkTitle :::", "");
  });
});
