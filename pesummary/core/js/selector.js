function setupPlotSelector(dropdownId, imageId, basePath, extension = '.png') {
    const dropdown = document.getElementById(dropdownId);
    const plotImage = document.getElementById(imageId);

    // Listen for the change event
    dropdown.addEventListener('change', function() {
        const selectedInput = this.value;
        // Construct the new source using the variable base path
        plotImage.src = `${basePath}${selectedInput}${extension}`;
    });
}

// Call the function for the different cases
setupPlotSelector('plotSelector', 'plotDisplay', '../plots/combined_jsd_plot_');
