    // Create Dino Constructor
    function Dino(dino) {
            this.species    = dino.species;
            this.weight     = Number(dino.weight);
            this.height     = Number(dino.height);
            this.diet       = dino.diet;
            this.when       = dino.when;
            this.where      = dino.where;
            this.fact       = dino.fact;
    }

    // Fetch Dinos from JSON file
    const fetchDinos = async () => {
      const response = await fetch('./dino.json');
      const DINOS = await response.json();
      const _DINOS = DINOS.map(dino => new Dino(dino));
      return _DINOS;
    }

    // Create Human Constructor
    function Human(name, height, weight, diet) {
      this.species = 'human';
      this.name = name;
      this.height = Number(height);
      this.weight = Number(weight);
      this.diet = diet;
    }

    // Use IIFE to get human data from form
    const getHumanData = (() => {
        const form = document.getElementById('dino-compare');
        form.addEventListener('submit', (event) => {
          event.preventDefault();
          const name    = document.getElementById('name').value;
          const feet    = document.getElementById('feet').value;
          const inches  = document.getElementById('inches').value;
          const weight  = document.getElementById('weight').value;
          const diet    = document.getElementById('diet').value;
      
          const heightInInches  = Number(feet) * 12 + Number(inches);
          const human           = new Human(name, heightInInches, weight, diet);
          console.log(human)
      
          generateInfographic(human);
        });
    })();

    // Create Dino Compare Method 1
    Dino.prototype.compareHeight = function(human) {
        if (this.height > human.height) {
          return `The ${this.species} was taller than you.`;
        } else if (this.height < human.height) {
          return `You are taller than the ${this.species}.`;
        } else {
          return `You are the same height as the ${this.species}.`;
        }
    };
      
    // Create Dino Compare Method 2
    Dino.prototype.compareWeight = function(human) {
        if (this.weight > human.weight) {
          return `The ${this.species} was heavier than you.`;
        } else if (this.weight < human.weight) {
          return `You are heavier than the ${this.species}.`;
        } else {
          return `You weigh the same as the ${this.species}.`;
        }
    };

    
    // Create Dino Compare Method 3
    Dino.prototype.compareDiet = function(human) {
        if (this.diet === human.diet) {
          return `You and the ${this.species} share the same diet.`;
        } else {
          return `Your diet is different from the ${this.species}.`;
        }
    };

  // To generate a random number between 0, 1, 2, 3, 4, and 6 
  const getRandomNumber = () => {
      const numbers = [0, 1, 2, 3, 4, 6];
      const randomIndex = Math.floor(Math.random() * numbers.length);
      return numbers[randomIndex];
  };

    // Generate Grid with randomized positions
  const generateGrid = (dinos, human) => {
    const gridContainer = document.getElementById('grid');
    gridContainer.innerHTML = '';
    const positions = [0, 1, 2, 3, 4, 5, 6, 7, getRandomNumber()];

    // Randomize the positions using Fisher-Yates algorithm
    for (let i = positions.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [positions[i], positions[j]] = [positions[j], positions[i]];
    }

    // Generate grid items based on positions
    for (let i = 0; i < positions.length; i++) {
        const position = positions[i];

        const item = document.createElement('div');
        item.className = 'grid-item';

        if (position === 4) {
            // Human tile
            item.innerHTML = `
                <h3>${human.name}</h3>
                <img src="images/human.png" alt="human">
            `;
        } else if (position === 7) {
            // Pigeon tile
            item.innerHTML = `
                <h3>${dinos[position].species}</h3>
                <img src="images/pigeon.png" alt="${dinos[position].species}">
                <p>All birds are dinosaurs.</p>
            `;
            console.log(item);
        } else {
            // Dinosaur tiles
            const dino = dinos[position];
            let fact;

            // Generate a random fact based on comparison methods
            const randomMethod = Math.floor(Math.random() * 3);
            if (randomMethod === 0) {
                fact = dino.compareHeight(human);
            } else if (randomMethod === 1) {
                fact = dino.compareWeight(human);
            } else {
                fact = dino.compareDiet(human);
            }

            item.innerHTML = `
                <h3>${dino.species}</h3>
                <img src="images/${dino.species.toLowerCase()}.png" alt="${dino.species}">
                <p>${fact}</p>
            `;
        }

        gridContainer.appendChild(item);
    }
  };

  // Generate Tiles for each Dino in Array
  const generateInfographic = async (human) => {
    const _DINOS = await fetchDinos();
    generateGrid(_DINOS, human);

    const form = document.getElementById('dino-compare');
    form.style.display = "none";

    const gridContainer = document.getElementById('grid');
    gridContainer.style.display = "grid";
  };
