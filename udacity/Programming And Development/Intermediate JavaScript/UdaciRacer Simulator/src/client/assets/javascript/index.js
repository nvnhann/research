// PROVIDED CODE BELOW (LINES 1 - 80) DO NOT REMOVE

// The store will hold all information needed globally
let store = {
  track_id: undefined,
  player_id: undefined,
  race_id: undefined,
};

// We need our javascript to wait until the DOM is loaded
document.addEventListener("DOMContentLoaded", function () {
  onPageLoad();
  setupClickHandlers();
});

async function onPageLoad() {
  try {
    getTracks().then((tracks) => {
      const html = renderTrackCards(tracks);
      renderAt("#tracks", html);
    });

    getRacers().then((racers) => {
      const html = renderRacerCars(racers);
      renderAt("#racers", html);
    });
  } catch (error) {
    console.log("Problem getting tracks and racers ::", error.message);
    console.error(error);
  }
}

function setupClickHandlers() {
  document.addEventListener(
    "click",
    function (event) {
      const { target } = event;

      // Race track form field
      if (target.matches(".card.track")) {
        handleSelectTrack(target);
      }

      // Podracer form field
      if (target.matches(".card.podracer")) {
        handleSelectPodRacer(target);
      }

      // Submit create race form
      if (target.matches("#submit-create-race")) {
        event.preventDefault();
        // start race
        handleCreateRace();
      }

      // Handle acceleration click
      if (target.matches("#gas-peddle")) {
        handleAccelerate();
      }
    },
    false
  );
}

async function delay(ms) {
  try {
    return await new Promise((resolve) => setTimeout(resolve, ms));
  } catch (error) {
    console.log("an error shouldn't be possible here");
    console.log(error);
  }
}
// ^ PROVIDED CODE ^ DO NOT REMOVE

// This async function controls the flow of the race, add the logic and error handling
async function handleCreateRace() {
  // render starting UI

  //Get player_id and track_id from the store
  const { player_id, track_id } = store;
  if(!player_id || !track_id) return;
  //invoke the API call to create the race, then save the result
  try {
    const race = await createRace(player_id, track_id);
    console.log("CREATE RACE::", race);
    // update the store with the race id
    store.race_id = race.ID;
    renderAt("#race", renderRaceStartView(race.Track, race.Cars));
    await runCountdown();
    console.log(`RUN COUNTDOWN::(${store.race_id})`);
    await startRace(store.race_id - 1);
	await runRace(store.race_id - 1)
  } catch (error) {
    console.log("ERROR::", error);
  }
}

async function runRace(raceID) {
  try {
    return await new Promise((resolve, reject) => {
      const racerInterval = setInterval(async () => {
        try {
          const getRaceResponse = await getRace(raceID); // Fetch the race data
          if (getRaceResponse.status === "in-progress") {
            renderAt("#leaderBoard", raceProgress(getRaceResponse.positions)); // Update the leaderboard with race progress
          } else if (getRaceResponse.status === "finished") {
            clearInterval(racerInterval); // Stop the interval if the race is finished
            renderAt("#race", resultsView(getRaceResponse.positions)); // Display the race results
            resolve(getRaceResponse); // Resolve the promise with the race response
          }
        } catch (error) {
          clearInterval(racerInterval); // Stop the interval if an error occurs
          reject(error); // Reject the promise with the error
        }
      }, 500);
    });
  } catch (error) {
    console.log("runRace error::", error); // Handle any errors that occur during the promise execution
  }
}

async function runCountdown() {
  try {
    await delay(1000); // Delay execution for 1 second

    let timer = 3; // Set initial timer value

    return new Promise((resolve) => {
      const countInterval = setInterval(() => {
        document.getElementById("big-numbers").innerHTML = --timer; // Update the timer display

        if (timer === 0) {
          clearInterval(countInterval); // Stop the countdown when timer reaches 0
          resolve(); // Resolve the promise to indicate countdown completion
        }
      }, 1000);
    });
  } catch (error) {
    console.log("runCountdown error::", error); // Handle any errors that occur during the countdown
  }
}

function handleSelectPodRacer(target) {
  console.log("selected a pod", target.id);

  // remove class selected from all racer options
  const selected = document.querySelector("#racers .selected");
  if (selected) {
    selected.classList.remove("selected");
  }

  // add class selected to current target
  target.classList.add("selected");

  //save the selected racer to the store
  store.race_id = target.id;
}

function handleSelectTrack(target) {
  console.log("selected a track", target.id);

  // remove class selected from all track options
  const selected = document.querySelector("#tracks .selected");
  if (selected) {
    selected.classList.remove("selected");
  }

  // add class selected to current target
  target.classList.add("selected");
  store.track_id = target.id;
  store.player_id = +target.id;
}

const handleAccelerate = async () => {
  console.log("accelerate button clicked");
  // Invoke the API call to accelerate
  await accelerate(store.race_id - 1);
};

// HTML VIEWS ------------------------------------------------
// Provided code - do not remove

function renderRacerCars(racers) {
  if (!racers.length) {
    return `
			<h4>Loading Racers...</4>
		`;
  }

  const results = racers.map(renderRacerCard).join("");

  return `
		<ul id="racers">
			${results}
		</ul>
	`;
}

function renderRacerCard(racer) {
  const { id, driver_name, top_speed, acceleration, handling } = racer;

  return `
		<li class="card podracer" id="${id}">
			<h3>${driver_name}</h3>
			<p>${top_speed}</p>
			<p>${acceleration}</p>
			<p>${handling}</p>
		</li>
	`;
}

function renderTrackCards(tracks) {
  if (!tracks.length) {
    return `
			<h4>Loading Tracks...</4>
		`;
  }

  const results = tracks.map(renderTrackCard).join("");

  return `
		<ul id="tracks">
			${results}
		</ul>
	`;
}

function renderTrackCard(track) {
  const { id, name } = track;

  return `
		<li id="${id}" class="card track">
			<h3>${name}</h3>
		</li>
	`;
}

function renderCountdown(count) {
  return `
		<h2>Race Starts In...</h2>
		<p id="big-numbers">${count}</p>
	`;
}

function renderRaceStartView(track, racers) {
  return `
		<header>
			<h1>Race: ${track.name}</h1>
		</header>
		<main id="two-columns">
			<section id="leaderBoard">
				${renderCountdown(3)}
			</section>

			<section id="accelerate">
				<h2>Directions</h2>
				<p>Click the button as fast as you can to make your racer go faster!</p>
				<button id="gas-peddle">Click Me To Win!</button>
			</section>
		</main>
		<footer class="bg-gray-900 py-4">
			<div class="container mx-auto text-center text-white">
				<p>Nguyễn Văn Nhẫn</p>
				<p><a href="mailto:NhanNV13@fpt.com">NhanNV13@fpt.com</a></p>
			</div>
		</footer>
	`;
}

function resultsView(positions) {
  positions.sort((a, b) => (a.final_position > b.final_position ? 1 : -1));

  return `
		<header>
			<h1>Race Results</h1>
		</header>
		<main>
			${raceProgress(positions)}
			<a class="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded" href="/race">Start a new race</a>
		</main>
	`;
}

function raceProgress(positions) {
  let userPlayer = positions.find((e) => e.id === store.player_id);
  userPlayer.driver_name += " (you)";

  positions = positions.sort((a, b) => (a.segment > b.segment ? -1 : 1));
  let count = 1;

  const results = positions.map((p) => {
    return `
			<tr>
				<td>
					<h3>${count++} - ${p.driver_name}</h3>
				</td>
			</tr>
		`;
  }).join("");

  return `
		<main>
			<h3>Leaderboard</h3>
			<section id="leaderBoard">
				${results}
			</section>
		</main>
	`;
}

function renderAt(element, html) {
  const node = document.querySelector(element);

  node.innerHTML = html;
}

// ^ Provided code ^ do not remove

// API CALLS ------------------------------------------------

const SERVER = "http://localhost:3001";

function defaultFetchOpts() {
  return {
    mode: "cors",
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": SERVER,
    },
  };
}

//Make a fetch call (with error handling!) to each of the following API endpoints

// ==================================== API TRACKS ==========================================
// getAll tracks
const getTracks = async () => {
  // request to `${SERVER}/api/tracks`
  try {
    const res = await fetch(`${SERVER}/api/tracks`, {
      method: "GET",
      dataType: "jsonp",
      mode: "cors",
      ...defaultFetchOpts(),
    });
    const data = await res.json();
    console.log("STRACKS::", data);
    return data;
  } catch (error) {
    console.log("Error occurred get TRACKS ==> : ", error);
  }
};

// ==========================================================================================

// ==================================== API RACES ============================================
// get all racers
const getRacers = async () => {
  // GET request to `${SERVER}/api/cars`
  try {
    const res = await fetch(`${SERVER}/api/cars`, {
      method: "GET",
      dataType: "jsonp",
      mode: "cors",
      ...defaultFetchOpts(),
    });
    const data = await res.json();
    console.log("CARS::", data);
    return data;
  } catch (error) {
    console.log("Error occurred get TRACKS ==> : ", error);
  }
};

// get race by id
const getRace = async (id) => {
  // GET request to `${SERVER}/api/races/${id}`
  try {
    const res = await fetch(`${SERVER}/api/races/${Number(id)}`, {
      method: "GET",
      dataType: "jsonp",
      mode: "cors",
      ...defaultFetchOpts(),
    });
    const data = await res.json();
    console.log(`TRACE by ${id}::`, data);
    return data;
  } catch (error) {
    console.log("Error occurred get RACE ==> : ", error);
  }
};

// create race
const createRace = async (player_id, track_id) => {
  player_id = parseInt(player_id);
  track_id = parseInt(track_id);

  const body = { player_id, track_id };

  try {
    const res = await fetch(`${SERVER}/api/races`, {
      method: "POST",
      ...defaultFetchOpts(),
      dataType: "jsonp",
      body: JSON.stringify(body),
    });

    console.log("CREATE RACE SUCCESSFULLY!");
    return await res.json();
  } catch (error) {
    console.log("Problem with createRace request::", error);
  }
};
// start race
const startRace = async (id) => {
  try {
    return await fetch(`${SERVER}/api/races/${Number(id)}/start`, {
      method: "POST",
      ...defaultFetchOpts(),
    });
  } catch (error) {
    console.log("Problem with getRace request::", error);
  }
};

// accelerate
const accelerate = async (id) => {
  // POST request to `${SERVER}/api/races/${id}/accelerate`
  // options parameter provided as defaultFetchOpts
  // no body or datatype needed for this request
  try {
   return await fetch(`${SERVER}/api/races/${Number(id)}/accelerate`, {
      method: "POST",
      ...defaultFetchOpts(),
    });
  } catch (error) {
    console.log("Problem with accelerate request::", error);
  }
};
