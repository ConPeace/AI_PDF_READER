function fetchBylaws() {
    fetch('/api/bylaws')
    .then(response => {
        if (response.ok) {
            return response.json();
        } else {
            throw new Error('Error: ' + response.statusText);
        }
    })
    .then(data => {
        let bylawContainer = document.getElementById('bylawContainer');
        for (let letter in data) {
            let letterDiv = document.createElement('div');
            letterDiv.innerHTML = `<h3>${letter}</h3>`;
            bylawContainer.appendChild(letterDiv);
            for (let bylaw of data[letter]) {
                let p = document.createElement('p');
                p.textContent = bylaw;
                p.className = 'bylaw';
                bylawContainer.appendChild(p);
                addBylawListener(p);
            }
        }
    })
    .catch(error => console.error('Error:', error));
}

function addBylawListener(element) {
    element.addEventListener('click', function() {
        document.getElementById('currentBylaw').textContent = 'You have selected ' + this.textContent;
    });
}

function submitQuestion() {
    let bylawType = document.getElementById('currentBylaw').innerText.split(' ').slice(3).join(' ').trim();
    let question = document.getElementById('userQuestion').value;
    let output = document.getElementById('output');

    fetch('/api', { 
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({bylaw: bylawType, question: question}),
    })
    
    .then(response => {
        if (response.ok) {
            return response.json();
        } else {
            throw new Error('Error: ' + response.statusText);
        }
    })
    .then(data => {
        output.innerText = 'Response: ' + data.answer
        console.log('Token usage: ' + data.token_usage)
    })
    .catch((error) => output.innerText = 'Error: ' + error);
}

window.onload = fetchBylaws;
