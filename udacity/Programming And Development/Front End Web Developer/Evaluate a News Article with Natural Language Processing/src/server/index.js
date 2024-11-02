require("dotenv").config();
const mockAPIResponse = require("./mockAPI.js");

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

const AylienNewsApi = require("aylien-news-api");
const defaultClient = AylienNewsApi.ApiClient.instance;

// Set the app_id and app_key correctly
defaultClient.authentications["app_id"].apiKey = process.env.API_APP_ID;
defaultClient.authentications["app_key"].apiKey = process.env.API_APP_KEY;

const api = new AylienNewsApi.DefaultApi();

// process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

app.use(express.static("src/client"));

app.get("/", function (req, res) {
  res.sendFile("/client/views/index.html", { root: __dirname + "/.." });
});

app.get("/test", (req, res) => {
  res.send(mockAPIResponse);
});

app.post("/api/getall", async (req, res) => {
  const { body } = req;
  const { title } = body;
  if (!title) return res.status(500).json({ errmsg: "Title is required!" });
  const opts = {
    title: "Bell",
    publishedAtStart: "NOW-7DAYS",
    publishedAtEnd: "NOW",
  };

  try {
    const data = await new Promise((resolve, reject) => {
      api.listStories(opts, (err, data) => {
        if (err) {
          reject(err);
        } else {
          resolve(data);
        }
      });
    });
    // .map((story) => ({
    //   title: story.title,
    //   source: story.source?.name,
    // })),
    return res.json({
      data: data.stories,
    });
  } catch (err) {
    return res.status(500).json({ err: err.message });
  }
});

// designates what port the app will listen to for incoming requests
app.listen(8080, () => {
  console.log("Example app listening on port 8080!");
});
