const baseURL = 'http://localhost:3000'
// Personal API Key for OpenWeatherMap API
const apiKey = 'ac7c6c1b6907928995f50837381506c2';

// Function to fetch data from the app endpoint
const retrieveData = async () => {
  const request = await fetch(`${baseURL}/all`);
  try {
    const allData = await request.json();
    console.log(allData);
    // Update DOM elements
    document.getElementById('temp').innerHTML = Math.round(allData.temp) + ' degrees';
    document.getElementById('content').innerHTML = allData.feelings;
    document.getElementById('date').innerHTML = allData.date;
  } catch (error) {
    console.log('error', error);
  }
};

// Event listener for the "generate" button
document.getElementById('generate').addEventListener('click', () => {
  const zipCode = document.getElementById('zip').value;
  const feelings = document.getElementById('feelings').value;
  const apiUrl = `https://api.openweathermap.org/data/2.5/weather?zip=${zipCode}&appid=${apiKey}`;

  fetch(apiUrl)
    .then(response => response.json())
    .then(data => {
      const temp = data.main.temp;
      const date = new Date().toLocaleDateString();
      const newData = { temp, date, feelings };

      fetch(`${baseURL}/addData`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(newData)
      })
        .then(response => response.json())
        .then(data => retrieveData())
        .catch(error => console.log('error', error));
    })
    .catch(error => console.log('error', error));
});
