var checkTitle = function (inputText) {
  console.log("::: Running checkTitle :::", inputText);
  if (!inputText) {
    alert("Title is required!");
    return false;
  }
  return true;
};

module.exports = {
  checkTitle: checkTitle,
};
