// Get the image and insert it inside the modal - use its "alt" text as a caption

function _onclick(img) {
    $(document).on('click', '#'+img, function(){$('#myModel').modal('show')});
}

function modal(id) {
    /* Show the modal when the image is clicked

    Parameters
    ----------
    id: str
        str giving the id of the clicked image
    */
    var img = document.getElementById(id);
    /*img.onclick = _onclick(id);*/
    _onclick(id)
}
