// server.js

// Require Express
const express = require("express");

// Create an instance of the Express app
const app = express();

// Require the cors package
const cors = require("cors");

// Require the body-parser package
const bodyParser = require("body-parser");

// Middleware
app.use(cors());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

// Set up the projectData object as the app API endpoint
let projectData = {};

// GET route to return the projectData object
app.get("/all", (_, res) => {
  res.send(projectData);
});

// POST route to add an entry to the projectData object
app.post("/addData", (req, res) => {
  const newData = req.body;
  projectData = {
    ...newData,
  };
  res.send(projectData);
});

// Start the server
const port = 3000;
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
