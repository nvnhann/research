-- Drop tables if they exist
DROP TABLE IF EXISTS "post_votes";
DROP TABLE IF EXISTS "comments";
DROP TABLE IF EXISTS "posts";
DROP TABLE IF EXISTS "topics";
DROP TABLE IF EXISTS "users";

-- Create Users table
CREATE TABLE "users" (
  "id" SERIAL PRIMARY KEY,
  "username" VARCHAR(25) UNIQUE NOT NULL,
  "login_timestamp" TIMESTAMP,
  CONSTRAINT "users_check" CHECK (LENGTH(TRIM("username")) > 0)
);

-- Indexes for Users table
CREATE INDEX "find_users_last_year_login" ON "users" ("username", "login_timestamp");
CREATE INDEX "find_user_by_username" ON "users" ("username");

-- Create Topics table
CREATE TABLE "topics" (
  "id" SERIAL PRIMARY KEY,
  "topic_name" VARCHAR(30) UNIQUE NOT NULL,
  "description" VARCHAR(500),
  "user_id" INTEGER,
  CONSTRAINT "topics_check" CHECK (LENGTH(TRIM("topic_name")) > 0)
);

-- Indexes for Topics table
CREATE INDEX "find_topics_no_posts" ON "topics" ("id");
CREATE INDEX "find_topic_by_name" ON "topics" ("topic_name");

-- Create Posts table
CREATE TABLE "posts" (
  "id" SERIAL PRIMARY KEY,
  "title" VARCHAR(100) NOT NULL,
  "url" TEXT,
  "text_content" TEXT,
  "user_id" INTEGER,
  "topic_id" INTEGER,
  "post_timestamp" TIMESTAMP,
  CONSTRAINT "posts_check" CHECK (LENGTH(TRIM("title")) > 0),
  CONSTRAINT "fk_user_post" FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE SET NULL,
  CONSTRAINT "fk_topic_post" FOREIGN KEY ("topic_id") REFERENCES "topics" ("id") ON DELETE CASCADE,
  CONSTRAINT "text_or_url_check" CHECK (("url" IS NULL AND "text_content" IS NOT NULL) OR ("url" IS NOT NULL AND "text_content" IS NULL))
);

-- Indexes for Posts table
CREATE INDEX "find_users_no_posts" ON "users" ("id");
CREATE INDEX "find_latest_posts_by_topic" ON "posts" ("topic_id", "post_timestamp");
CREATE INDEX "find_latest_posts_by_user" ON "posts" ("user_id", "post_timestamp");
CREATE INDEX "find_posts_with_URL" ON "posts" ("url");

-- Create Comments table
CREATE TABLE "comments" (
  "id" SERIAL PRIMARY KEY,
  "comment" TEXT NOT NULL,
  "user_id" INTEGER,
  "topic_id" INTEGER,
  "post_id" INTEGER,
  "parent_comment_id" INTEGER DEFAULT NULL,
  CONSTRAINT "comments_check" CHECK (LENGTH(TRIM("comment")) > 0),
  CONSTRAINT "fk_user_comment" FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE SET NULL,
  CONSTRAINT "fk_topic_comment" FOREIGN KEY ("topic_id") REFERENCES "topics" ("id") ON DELETE CASCADE,
  CONSTRAINT "fk_post_comment" FOREIGN KEY ("post_id") REFERENCES "posts" ("id") ON DELETE CASCADE,
  CONSTRAINT "comment_thread" FOREIGN KEY ("parent_comment_id") REFERENCES "comments" ("id") ON DELETE CASCADE
);

-- Indexes for Comments table
CREATE INDEX "top_level_comments" ON "comments" ("post_id", "parent_comment_id") WHERE "parent_comment_id" IS NULL;
CREATE INDEX "direct_children_comments" ON "comments" ("parent_comment_id");
CREATE INDEX "latest_comments_by_user" ON "comments" ("user_id");

-- Create Post Votes table
CREATE TABLE "post_votes" (
  "id" SERIAL PRIMARY KEY,
  "post_vote" INTEGER NOT NULL,
  "voter_user_id" INTEGER,
  "post_id" INTEGER,
  CONSTRAINT "vote_check" CHECK ("post_vote" = 1 OR "post_vote" = -1),
  CONSTRAINT "fk_user_vote" FOREIGN KEY ("voter_user_id") REFERENCES "users" ("id") ON DELETE SET NULL,
  CONSTRAINT "fk_post_vote" FOREIGN KEY ("post_id") REFERENCES "posts" ("id") ON DELETE CASCADE
);

-- Indexes for Post Votes table
CREATE INDEX "post_scores" ON "post_votes" ("post_vote", "post_id");

-- Insert unique usernames from both initial tables into "users"
INSERT INTO "users" ("username")
SELECT DISTINCT username
FROM (
    SELECT username FROM "bad_posts"
    UNION
    SELECT username FROM "bad_comments"
    UNION
    SELECT UNNEST(string_to_array(upvotes, ',')) AS username FROM "bad_posts"
    UNION
    SELECT UNNEST(string_to_array(downvotes, ',')) AS username FROM "bad_posts"
) AS unique_usernames;



-- Insert distinct topics from "bad_posts" into "topics"
INSERT INTO "topics" ("topic_name")
SELECT DISTINCT topic
FROM "bad_posts";

-- Insert fields from "bad_posts", "users", and "topics" into "posts"
INSERT INTO "posts" ("title", "url", "text_content", "user_id", "topic_id")
SELECT LEFT(bp.title, 100), bp.url, bp.text_content, u.id AS user_id, tp.id AS topic_id
FROM "bad_posts" bp
JOIN "users" u ON bp.username = u.username
JOIN "topics" tp ON bp.topic = tp.topic_name;

-- Insert comments and IDs into "comments"
INSERT INTO "comments" ("comment", "user_id", "topic_id", "post_id")
SELECT bc.text_content AS comment, p.user_id, p.topic_id, p.id AS post_id
FROM "bad_comments" bc
JOIN "users" u ON bc.username = u.username
JOIN "posts" p ON bc.post_id = p.id;

-- Insert upvotes & downvotes into "post_votes"
INSERT INTO "post_votes" ("post_vote", "voter_user_id", "post_id")
SELECT post_vote, u.id, p.id
FROM (
    SELECT 1 AS post_vote, upvote AS username, id
    FROM "bad_posts", UNNEST(string_to_array(upvotes, ',')) AS upvote
    UNION ALL
    SELECT -1, downvote, id
    FROM "bad_posts", UNNEST(string_to_array(downvotes, ',')) AS downvote
) AS votes
JOIN "users" u ON votes.username = u.username
JOIN "posts" p ON votes.id = p.id;
