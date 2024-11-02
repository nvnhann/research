// titleChecker.test.js
var checkTitle = require("../src/client/js/titleChecker").checkTitle;

test("checkTitle should return false for invalid titles", () => {
  expect(checkTitle("")).toBe(false);
  expect(checkTitle("1234567890123456789012345678901")).toBe(true);
  // Add more test cases for invalid titles
});
