let store = Immutable.Map({
  user: Immutable.Map({ name: "Student" }),
  apod: "",
  rovers: Immutable.List([
    "Curiosity",
    "Opportunity",
    "Spirit",
    "Perseverance",
  ]),
  current: "",
});

// add our markup to the page
const root = document.getElementById("root");

// update store
const updateStore = (store, newState) => {
  store = Object.assign(store, newState);
  render(root, store);
};

// render html
const render = async (root, state) => {
  root.innerHTML = App(state);
};

// create content
const App = (state) => {
  const header = `
                  <header>
                    <nav class="bg-gray-800">
                    <div class="container mx-auto px-4 css1">
                      <div class="flex items-center justify-center h-16 css2">
                        <div class="flex items-center">
                          <a href="#" class="text-white text-lg font-semibold" onclick="handleReset()">NhanNV13</a>
                        </div>
                        <div class="flex items-center menu-item .css3">
                          ${MenuItems(state)}
                        </div>
                      </div>
                    </div>
                  </nav>
                  </header>`;

  const footer = `<footer>
    <div class="footer-container">
      <p class="name">Nguyễn Văn Nhẫn</p>
      <p class="email">NhanNV13@fpt.com</p>
    </div>
  </footer>`;
  if (!state.get("current")) {
    return `
        ${header}
        <main class="h-screen">
            <section>
             ${ImageOfTheDay(state)}
            </section>
        </main>
        ${footer}
        
    `;
  } else {
    return `
    ${header}
      <main class="h-screen">
        <section id="section" class="grid gap-4">
          ${renderImageRecover(state)}
        </section>
      </main>
    ${footer}

    `;
  }
};

// listening for load event because page should load before any JS is called
window.addEventListener("load", () => {
  render(root, store);
});

// Example of a pure function that renders infomation requested from the backend
const ImageOfTheDay = (state) => {
  // If image does not already exist, or it is not from today -- request it again
  const apod = state.get("apod");
  if (!apod) {
    getImageOfTheDay(store);
  }

  // check if the photo of the day is actually type video!
  if (apod.media_type === "video") {
    return `
          <p>See today's featured video <a href="${apod.url}">here</a></p>
          <p>${apod.title}</p>
          <p>${apod.explanation}</p>
      `;
  } else {
    return `
          <img src="${apod.url}" height="350px" width="100%" />
          <p>${apod.explanation}</p>
      `;
  }
};

// onclick set cover to current
const handleClick = async (event) => {
  const { id } = event.currentTarget;
  console.log(id);
  if (!!id) {
    await getImageRecover(store, id);
  }
};
// On Click reset home page
const handleReset = () => {
  updateStore(store, store.set("current", ""));
};

// render menu item
const MenuItems = (state) => {
  return state
    .get("rovers")
    .map(
      (item) => `
    <a href="#${item}" id="${item}" class="text-gray-300 hover:text-white px-3 py-2" onclick="handleClick(event)">
      ${item}
    </a>`
    )
    .join("");
};

// ------------------------------------------------------  API CALLS

// Example API call
const getImageOfTheDay = async (state) => {
  let { apod } = state;
  const res = await fetch(`http://localhost:3000/apod`);
  apod = await res.json();

  updateStore(store, store.set("apod", apod.data));
  return apod;
};

const renderImageRecover = (state) => {
  return state
    .get("current")
    .latest_photos.map(
      (item) => `
                <div class="rounded overflow-hidden shadow-lg">
                <img class="w-full" src="${item.img_src}" alt="Rover Image">
                <div class="px-6 py-4">
                  <div class="font-bold text-xl mb-2">Image Date: ${item.earth_date}</div>
                  <p><span class="font-bold">Rover:</span> ${item.rover.name}</p>
                  <p><span class="font-bold">State of the Rover:</span> ${item.rover.status}</p>
                  <p><span class="font-bold">Launch Date:</span> ${item.rover.launch_date}</p>
                  <p><span class="font-bold">Landing Date:</span> ${item.rover.landing_date}</p>
                </div>
              </div>
          `
    )
    .join("");
};

// call API get cover from name
const getImageRecover = async (store, name) => {
  const res = await fetch(`http://localhost:3000/rovers/${name}`);
  const current = await res.json();
  updateStore(store, store.set("current", current));
  return current;
};
