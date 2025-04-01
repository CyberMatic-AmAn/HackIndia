function updateTrafficData() {
    fetch('/traffic_data')
        .then(response => response.json())
        .then(data => {
            const laneData = document.getElementById('lane-data');
            laneData.innerHTML = '';
            for (let i = 0; i < 4; i++) {
                const laneDiv = document.createElement('div');
                laneDiv.innerHTML = `
                    <p>Lane ${i + 1}: 
                        Vehicles: <span>${data.total_vehicles[i]}</span>, 
                        Density: <span>${data.current_density[i]}</span>, 
                        Avg Density: <span>${data.avg_density[i]}</span>, 
                        Light: <span class="light" style="color: ${data.traffic_lights[i] === 'Green' ? 'green' : data.traffic_lights[i] === 'Yellow' ? 'yellow' : 'red'}">${data.traffic_lights[i]}</span>
                    </p>`;
                laneData.appendChild(laneDiv);
            }
            document.getElementById('timestamp').textContent = data.timestamp;
        })
        .catch(error => console.error('Error fetching traffic data:', error));
}

// Update data every 2 seconds
setInterval(updateTrafficData, 2000);

// Initial update
updateTrafficData();