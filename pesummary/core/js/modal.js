// Get the image and insert it inside the modal - use its "alt" text as a caption

function _onclick(img) {
    $(document).on('click', '#'+img, function(){$('#myModel').modal('show')});
    $(document).keydown(function(e) {
      if (e.keyCode === 37) {
        $(".carousel-control-prev").click();
        return false;
      }
      if (e.keyCode === 39) {
        $(".carousel-control-next").click();
        return false;
      }
    });
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

function changeimage(id) {
    /* */
    var img = document.getElementById(id);
    var current_src = img.src;
    if ( current_src.indexOf("posterior") >= 0 ) {
        img.src = current_src.replace("1d_posterior", "cdf");
    }
    else if ( current_src.indexOf("cdf") >= 0 ) {
        img.src = current_src.replace("cdf", "1d_posterior");
    }
}
