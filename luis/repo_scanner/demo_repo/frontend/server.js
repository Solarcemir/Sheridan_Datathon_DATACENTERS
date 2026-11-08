/* Fake demo frontend -- NOT a real app, exists only for luis/repo_scanner's demo_repo. Do not run this;
   it is deliberately insecure in the exact ways its pinned dependencies (package-lock.json) are. */
const express = require("express");
const _ = require("lodash");
const minimist = require("minimist");
const fetch = require("node-fetch");

const args = minimist(process.argv.slice(2));
const app = express();

app.get("/greet", (req, res) => {
  // _.template on user input -- lodash template-injection CVEs (e.g. CVE-2021-23337) are exactly this.
  const compiled = _.template(req.query.name || "world");
  res.send("Hello, " + compiled({}));
});

app.get("/fetch", async (req, res) => {
  const r = await fetch(req.query.url);
  res.json({ status: r.status });
});

app.listen(args.port || 3000);
