require("dotenv").config();
const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");

const fetch = require("node-fetch");
const path = require("path");

const app = express();
app.use(cors());
const port = 3000;

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

app.use("/", express.static(path.join(__dirname, "../public")));

// Get rover by name
app.get("/rovers/:name", async (req, res) => {
  try {
    const response = await fetch(
      `https://api.nasa.gov/mars-photos/api/v1/rovers/${req.params.name}/latest_photos?api_key=${process.env.API_KEY}`
    );
    const data = await response.json();
    res.send(data);
  } catch (err) {
    console.log("error:", err);
  }
});

// example API call
app.get("/apod", async (req, res) => {
  try {
    const response = await fetch(
      `https://api.nasa.gov/planetary/apod?api_key=${process.env.API_KEY}`
    );
    const data = await response.json();
    res.send({ data });
  } catch (err) {
    console.log("error:", err);
  }
});

app.listen(port, () => console.log(`Example app listening on port ${port}!`));
