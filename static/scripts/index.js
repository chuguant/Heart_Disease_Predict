$(() => {
    $("#submit-spinner").hide();

    $("#submit-btn").click(() => {

        $("#submit-spinner").show();
        $("#submit-btn").addClass("disabled");

        value = []
        for (i of Array(13).keys()) {
            value.push(parseFloat($(`#input_${i}`).val()))
        }

        var options = {
            url: `/api/predict`,
            method: 'POST',
            data: {
                value: value,
            },
            headers: {
                'content-type': 'application/json'
            }
        }

        axios(options)
            .then((response) => {
                console.log(response);
                if (response.data['has_disease'] == true) {
                    $("#result").text("Have heart disease");
                } else {
                    $("#result").text("No heart disease");
                }
            })
            .catch((error) => {
                alert('Server unavailable!');
                console.log(error);
                $("#submit-spinner").hide();
                $("#submit-btn").removeClass("disabled");
            })
            .then(function () {
                $("#submit-spinner").hide();
                $("#submit-btn").removeClass("disabled");
            });
    })
})
