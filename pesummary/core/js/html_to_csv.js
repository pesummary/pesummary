/*Script taken from https://www.codexworld.com/export-html-table-data-to-csv-using-javascript/ */

function download_csv(csv, filename) {
    var csvFile;
    var downloadLink;

    csvFile = new Blob([csv], {type: "text/csv"});
    downloadLink = document.createElement("a");
    downloadLink.download = filename;
    downloadLink.href = window.URL.createObjectURL(csvFile);
    downloadLink.style.display = "none";
    document.body.appendChild(downloadLink);

    downloadLink.click();
}

function export_table_to_csv(filename) {
	var csv = [];
	var rows = document.querySelectorAll("table tr");
	
    for (var i = 0; i < rows.length; i++) {
		var row = [], cols = rows[i].querySelectorAll("td, th");
		
        for (var j = 0; j < cols.length; j++) 
            row.push(cols[j].innerText);
        
		csv.push(row.join(","));		
	}

    download_csv(csv.join("\n"), filename);
}

function export_table_to_pip(filename) {
var csv = [];
        var rows = document.querySelectorAll("table tr");

    for (var i = 1; i < rows.length; i++) {
                var row = [], cols = rows[i].querySelectorAll("td, th");

        for (var j = 0; j < cols.length; j++)
            row.push(cols[j].innerText);

                csv.push(row.join("=="));
        }

    download_csv(csv.join("\n"), filename);
}

function export_table_to_conda(filename) {
    var csv = [];
    var pypi = [];
    var rows = document.querySelectorAll("table tr");

    csv.push("name: pesummary")
    csv.push("channels:")
    csv.push("- conda-forge")
    csv.push("dependencies:")
    for (var i = 1; i < rows.length; i++) {
        var row = [], cols = rows[i].querySelectorAll("td, th");

        if (cols[2].innerText === "pypi") {
            row.push(cols[0].innerText)
            row.push(cols[1].innerText)
            pypi.push(row.join("="))
        } else {
            for (var j = 0; j < cols.length; j++) {
                if (j != 2) {
                    row.push(cols[j].innerText);
                }
            }
            csv.push("- " + row.join("="));
        }
    }
    if (pypi.length != 0) {
        csv.push("- pip:")
        for (var j = 0; j < pypi.length; j++) {
            csv.push("  - " + pypi[j])
        }
    }
    download_csv(csv.join("\n"), filename);
}
