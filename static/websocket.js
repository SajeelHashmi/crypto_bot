


    const socket = io();
    // const klineSocket = io('wss://stream.binancefuture.com/ws');
    let kline_id = 1;



// start stop test buttons

document.getElementById('startTest').addEventListener('click',()=>{
    fetch('/startTest', {
        method: 'GET',
    }).then(res => res.json()).then(data => {
        
        console.log(data)
        
    })
})

document.getElementById('stopTest').addEventListener('click',()=>{
    fetch('/stopTest', {
        method: 'GET',
    }).then(res => res.json()).then(data => {
        
        console.log(data)
        
    })
})





    socket.on('signal', (data) => {
        const key = data['key'];
        const coin = key.split('_')[0];
        const timeframe = key.split('_')[1];
        // console.log(key)
        const row =  document.querySelector(`tr[data-key="${key}"]`);
        if (row){
            const trend =  data['trend'];
            let trendStr = '';
            if (trend == 0){
                trendStr = 'Sideways';
            }
            else if(trend == 1){
                trendStr = 'Uptrend';
            }
            else if (trend == -1){
                trendStr = 'Downtrend';
            }
            row.children[2].innerText = trendStr;
            const signals = data['signals'];
            for (let index = 0; index < signals.length; index++) {
                const element = signals[index];
                let signal = '';
                let cl = "text"
                if (element == 0){
                    signal = 'No signal';
                
                }
                else if(element == 1){
                    // window.alert(`Long Signal for symbol ${coin}`)
                    signal = 'Long';
                    cl = "btn-primary"
                }
                else if (element == -1){
                    // window.alert(`Short Signal for symbol ${coin}`)

                    signal = 'Sell';
                    cl = "btn-danger"
                }
                row.children[index+3].innerText = signal;
                row.children[index+3].classList.add(cl);
                // console.log(data,index+3)
            }
        }
        else{
            createStrategyRow(key,coin,timeframe);
        }
    });

    const logoutBtn = document.getElementById('logoutBtn');
    logoutBtn.addEventListener('click', () => {
        fetch('/logout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        }).then(res => res.json()).then(data => {
            if ('msg' in data){
                if (data['msg'] == 'Logged Out'){
                    socket.disconnect()
                    window.location.href = '/';
                }
            }
        })
    });


    const stratBtn = document.getElementById('startStratBtn');
    stratBtn.addEventListener('click', () => {
        const coin = document.getElementById('coin').value;
        const timeframe = document.getElementById('timeframe').value;
        fetch('/startstrategy', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ coin, timeframe })
        }).then(res => res.json()).then(data => {
            if ('key' in data){
                const key = data['key'];
                console.log(key);
                const row = document.querySelector(`tr[data-key="${key}"]`);
                if (row){
                    return;
                }
                else{
                    createStrategyRow(key,coin,timeframe);
                }

            }
        })
        socket.emit('tradeSignal');

});
socket.on('tradeSignal', (data)=> {
    const symbol = data['contract'];

    let s = symbol.toLowerCase()
    const row =  document.querySelector(`#tradeTable tr[data-symbol="${s}"]`);
    // if(data['status']!='open'){
    //     row.remove()
    //     window.alert(`Closed Position for ${symbol} Using ${data['strategy']} Strategy With Pnl ${data['pnl']}`)
    // }
    if (row){
        row.children[0].innerText = symbol;
        
        const strategy =  data['strategy'];
        row.children[1].innerText = strategy;

        const side =  data['side'];
        row.children[2].innerText = side;

        const enter =  data['entry'];
        row.children[3].innerText = enter;

        const currentPrice =  data['currentPrice'];
        row.children[4].innerText = currentPrice;

        const pnl =  data['pnl'];
        row.children[5].innerText = pnl;

    }
    else{
        createTradeRow(data);
    }
});


    function createStrategyRow(key,coin,timeframe){

        const table = document.querySelector('#strategyTable tbody');
                    const newRow = document.createElement('tr');
                    newRow.setAttribute('data-key', key);
                    newRow.innerHTML = `
                    <td>${coin}</td>
                    <td>${timeframe}</td>

                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>
                    <td> <button class = "btn"> wating for data </button> </td>

                    `;
                    table.appendChild(newRow);
                    const button = document.createElement('button');
                    button.setAttribute('data-key', key);
                    button.classList.add('stopStrat');
                    button.classList.add('btn');
                    button.classList.add('btn-danger');
                    button.innerText = 'Stop';
                    button.addEventListener('click', (event) => {
                        const key = event.target.getAttribute('data-key');
                        fetch('/stopstrategy', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ key })
                        }).then(res => res.json()).then(data => {
                            if ('status' in data){
                                if (data['status'] == 'success'){
                                    newRow.remove();
                                }
                            }
                        })
                    });
                    const td = document.createElement('td');
                    td.appendChild(button);
                    newRow.appendChild(td);

    }
  


    function createTradeRow(data){
        const symbol = data['contract']
        let s = symbol.toLowerCase()
        const table = document.querySelector('#tradeTable tbody');
                    const newRow = document.createElement('tr');
                    newRow.setAttribute('data-symbol', s);
                    newRow.innerHTML = `
                    <td>${symbol}</td>

                    <td>${data['strategy']}</td>

                    <td>${data['side']}</td>

                    <td>${String(data['entry'])}</td>

                    <td>${data['currentPrice']}</td>

                    <td>${data['pnl']}</td>
                    `;
                    table.appendChild(newRow);
                    const button = document.createElement('button');
                    button.setAttribute('data-symbol', data['contract']);
                    button.setAttribute('data-strat', data['strategy']);

                    button.classList.add('closePosition');
                    button.classList.add('btn');
                    button.classList.add('btn-danger');
                    button.innerText = 'Close Position';
                    button.addEventListener('click', (event) => {
                        const symbol = event.target.getAttribute('data-symbol');
                        const strat = event.target.getAttribute('data-strat');
                        console.log('close position called',strat,symbol)
                        fetch('/closeposition', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                "symbol": symbol,
                                "strategy": strat
                            })
                        }).then(res => res.json()).then(data => {
                            if (data['status'] == 'Trade closed successfully'){
                                newRow.remove();
                            }
                        })                        
                    });
                    const td = document.createElement('td');
                    td.appendChild(button);
                    newRow.appendChild(td);
                    // Timeframe doesnot matter as we are only interested in close of each candle
          
    }
  
    // klineSocket.addEventListener('message',  (event)=> {
    //     console.log('Message from TESTNET:', event.data);
    // });






















// var socket = new WebSocket("wss://stream.binancefuture.com/ws");

// // Event handler for when the WebSocket connection is opened
// socket.onopen = function(event) {
//     console.log("WebSocket connection opened");
//     var subscribeMessage = {
//         method: "SUBSCRIBE",
//         params: ["btcusdt@aggTrade"],
//         id: 1
//     };
//     socket.send(JSON.stringify(subscribeMessage));
// };

// // Event handler for when a message is received from the server
// socket.onmessage = function(event) {
//     console.log("Message received:", event.data);
//     // Handle the received message here
// };

// // Event handler for when an error occurs
// socket.onerror = function(error) {
//     console.error("WebSocket error:", error);
// };

// // Event handler for when the WebSocket connection is closed
// socket.onclose = function(event) {
//     console.log("WebSocket connection closed");
// };