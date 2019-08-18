var el = x => document.getElementById(x);
var response

function renderImage(img){
    if ((typeof img) == "number") return 
    grid = el("grids")
    if (img.startsWith("data:")){
        var tag = document.createElement('img');
        tag.src = img
    } else {
        var tag = document.createElement('p')
        tag.id = img
    }
    var gdiv = document.createElement('div');
    gdiv.className = "grid-item"
    gdiv.appendChild(tag)
    grid.appendChild(gdiv)
}

function showPicker() {
    el("file-input").click();
}

function showPicked(input) {
    el("upload-label").innerHTML = input.files[0].name;
    var reader = new FileReader();
    reader.onload = function (e) {
        el("image-picked").src = e.target.result;
        el("image-picked").className = "";
    };
    reader.readAsDataURL(input.files[0]);
}

function analyze() {
    var uploadFiles = el("file-input").files;
    if (uploadFiles.length !== 1) alert("Please select a file to analyze!");

    el("analyze-button").innerHTML = "Analyzing...";
    var xhr = new XMLHttpRequest();
    var loc = window.location;
    xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`,true);
    xhr.onerror = function () {
        alert(xhr.responseText);
    };
    xhr.onload = function (e) {
        if (this.readyState === 4) {
            response = JSON.parse(e.target.responseText)
            var cams = response["cam"]
            el("grids").innerHTML=""
            for(ima of cams){
                ima.map(renderImage)
            }
            var codes = response["result"]
            el("result-label").innerHTML = `<ul id="result-texts" style="list-style: none;"></ul>`
            var txtprms = codes.map(
                y => fetch("http://iconclass.org/json/?notation=" + y[0])
                    .then(x => x.json())
                    .then(x => x[0].txt.en)
                    .then(function(x){
                        console.log(x)
                        /*var li = document.createElement('li')
                        li.innerText = x
                        el("result-texts").appendChild(li)*/
                        el(y[0]).innerHTML = `${x} 
                        <bold>( ${Number.parseFloat(y[1]*100).toFixed(2)}% )</bold>`
                    }))
            Promise.all(txtprms)
        }
        el("analyze-button").innerHTML = "Analyze";
    };

    var fileData = new FormData();
    fileData.append("file", uploadFiles[0]);
    xhr.send(fileData);
}
