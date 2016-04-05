	// Global Variables
	var scrollRate = 1000;
	var imageDirectory;
	var globalData;

	var frameWidth = 500.0;
	var numDominantColours;

    //var $slider = $('#cur-pos');
    var $slider = $('#cur-pos');
    var $scrollBar = $('#scroll-bar');

	var QueryString = function () {
	  // This function is anonymous, is executed immediately and
	  // the return value is assigned to QueryString!
	  var query_string = {};
	  var query = window.location.search.substring(1);
	  var vars = query.split("&");
	  for (var i=0;i<vars.length;i++) {
		var pair = vars[i].split("=");
			// If first entry with this name
		if (typeof query_string[pair[0]] === "undefined") {
		  query_string[pair[0]] = decodeURIComponent(pair[1]);
			// If second entry with this name
		} else if (typeof query_string[pair[0]] === "string") {
		  var arr = [ query_string[pair[0]],decodeURIComponent(pair[1]) ];
		  query_string[pair[0]] = arr;
			// If third or later entry with this name
		} else {
		  query_string[pair[0]].push(decodeURIComponent(pair[1]));
		}
	  }
		return query_string;
	}();

    function SVG(tag) {
       return document.createElementNS('http://www.w3.org/2000/svg', tag);
    }

	$(function() {
		var videoName = QueryString.video;
		console.log('Video Name: ', videoName);
		$('#video-title').text(videoName);

        //TODO add buttons to fly through frames and stop at certain scene points. easy enough with scroll?

        $scrollBar.click(function(e) {
            //todo make sure click below isnt called?
		    var x = e.pageX - this.offsetLeft;
			var y = e.pageY - this.offsetTop;
            x -= document.body.scrollLeft;
            $slider.attr('x', x);
            //$slider.attr('transform', "translate(" + x + ", 0)");
            var fraction = x / 500.0;
            document.body.scrollLeft = Math.round(fraction * document.body.scrollWidth);
		});

        $slider.draggable().bind('mousedown', function(event, ui) {
            $scrollBar.css('opacity', 0.7);
            // bring target to front
            $(event.target.parentElement).append( event.target );
          })
          .bind('drag', function(event, ui){
              // update coordinates manually, since top/left style props don't work on SVG
              event.target.setAttribute('x', ui.position.left);
              //event.target.setAttribute('transform', "translate(" + ui.position.left + ", 0)");

              var fraction = $slider.attr('x') / 500.0;
              //var s = $slider.attr('transform').split(',')[0].split('(')[1];
              //console.log(s);
              //var fraction = ui.position.left / 500.0;
              document.body.scrollLeft = fraction * document.body.scrollWidth;
          }).bind('mouseup', function(event, ui) {
            $scrollBar.css('opacity', 0.8); //todo good?
        });

        $(document).scroll(function(e) {
            var fraction = this.body.scrollLeft / this.body.scrollWidth;
            var distx = fraction * 500.0;

            distx = Math.round(distx).toString();
            $slider.attr('x', distx);
        });

        $("body").mousewheel(function(event, delta) {
			this.scrollLeft -= (delta * scrollRate);

			event.preventDefault();
		});

		imageDirectory = 'example_images/' + videoName + '/images/full/';
		loadDataset(videoName);
	});

	// Data Loading
    function loadDataset(video_name) {
		var url = "/get_json/" + video_name;
		$.get(url, function(data) {
			if (!data) {
				console.log('Failure getting json');
				return;
			}
			console.log(data);
            console.log("Number of Images:", data.images.length);
			globalData = data;
			var $pageWrap = $("#page-wrap");
			numDominantColours = data.images[0].dominant_colours.kmeans.length;

			Handlebars.registerHelper('divide', function (num1, num2) {
				// CONVERT TO TIME TODO
				return (num1 / num2).toFixed(3);
			});

            Handlebars.registerHelper('toFixed', function(num, degrees) {
               return num.toFixed(degrees);
            });

            Handlebars.registerHelper('secondsToMinutes', function(seconds) {
                seconds = Math.round(seconds);
                var m = Math.floor(seconds / 60);
                var s = (seconds - m * 60).toString();
                if (s.length === 1) s = "0" + s;
                return m + ":" + s;
            });

            Handlebars.registerHelper('frameNoToTime', function(frameNo) {
                var frac = frameNo / data.info.framecount;

                var seconds = frac * Math.round(data.info.length);
                seconds = Math.round(seconds);
                var m = Math.floor(seconds / 60);
                var s = (seconds - m * 60).toString();
                if (s.length === 1) s = "0" + s;
                return m + ":" + s;
            });

			Handlebars.registerHelper('addDirectory', function(image_name) {
				return imageDirectory + image_name;
			});

			Handlebars.registerHelper('divide', function(frame_number) {
				return (frame_number / data.info.fps).toFixed(2);
			});

			Handlebars.registerHelper('ifCond', function (v1, operator, v2, options) {

				switch (operator) {
					case '==':
						return (v1 == v2) ? options.fn(this) : options.inverse(this);
					case '===':
						return (v1 === v2) ? options.fn(this) : options.inverse(this);
					case '<':
						return (v1 < v2) ? options.fn(this) : options.inverse(this);
					case '<=':
						return (v1 <= v2) ? options.fn(this) : options.inverse(this);
					case '>':
						return (v1 > v2) ? options.fn(this) : options.inverse(this);
					case '>=':
						return (v1 >= v2) ? options.fn(this) : options.inverse(this);
					case '&&':
						return (v1 && v2) ? options.fn(this) : options.inverse(this);
					case '||':
						return (v1 || v2) ? options.fn(this) : options.inverse(this);
					default:
						return options.inverse(this);
				}
			});

            Handlebars.registerHelper('subtract', function (a, b) {
                return a - b;
			});

			Handlebars.registerHelper('getColourString', function (a, b, c) {
				return Math.round(a) + ", " +  Math.round(b) + ", " + Math.round(c);
			});

			widthTravelled = 0;

			Handlebars.registerHelper('getStartRect', function (count, index) {
				var width = (count / 10000.0) * frameWidth;

				var returnValue = widthTravelled;
				widthTravelled += width;

				if (index == (numDominantColours - 1)) {
					widthTravelled = 0;
				}
				return Math.round(returnValue);
			});

			Handlebars.registerHelper('getRectWidth', function (count) {
				var width = (count / 10000.0) * frameWidth;

				return Math.round(width);
			});

			Handlebars.registerHelper('getFrameWidth', function () {
				return frameWidth;
			});

			var theTemplateScript = $("#image-template").html();
			var theTemplate = Handlebars.compile(theTemplateScript);
			var theCompiledHtml = theTemplate(data);
			$pageWrap.append(theCompiledHtml);

            // todo bright flashy colours to indicate scene change
			var lastSceneNumber = 0;
			$pageWrap.find('td').each(function(index) {
				var $this = $(this);
				var this_scene_number = parseInt($this.find('.scene_number_span').text());
				if (lastSceneNumber !== this_scene_number || index === 0) {
                    var theTemplateScript = $("#scene-change-template").html();
                    var theTemplate = Handlebars.compile(theTemplateScript);
                    var theCompiledHtmljQ = $(theTemplate(data.scenes[this_scene_number])); //this or last

                    theCompiledHtmljQ.insertBefore($this);
                    lastSceneNumber = this_scene_number;
				}
			});

            var headerScript = $("#video-info-template").html();
			var headerTemplate = Handlebars.compile(headerScript);
			var compiledHtml = headerTemplate(data);
            $(compiledHtml).insertBefore($('td').first());

            if (data.scenes) {
                var rgbString = 'rgb(' + Math.round(data.scenes[0].average_colour[0]) + ', ' + Math.round(data.scenes[0].average_colour[1]) + ', ' + Math.round(data.scenes[0].average_colour[2]) + ')';
                $('#video-title').css('color', rgbString);
            }

            // All gif code
            var imageFromEachScene = [];
            var NUM_IMAGES_PER_SCENE = data.info.INITIAL_NUM_FRAMES_IN_SCENE || 5; //TODO change in future

            // seems very fragile todo
            for (var i = NUM_IMAGES_PER_SCENE - 2; i < data.images.length; i += NUM_IMAGES_PER_SCENE) {
                imageFromEachScene.push(imageDirectory + data.images[i].image_name);
            }

            var numGifImages = imageFromEachScene.length;
            var currImageIndex = 0;
            var $gif = $('#gif');
            var $gifSceneNum = $('#gif-scene-num');
            $gif.attr('src', imageFromEachScene[currImageIndex++]);

            var gifChangeSpeed = 400;

            var gifInterval;
            $gif.mouseover(function() {
                gifInterval = setInterval(function () {
                    $gifSceneNum.text(currImageIndex); //todo

                    $gif.attr('src', imageFromEachScene[currImageIndex++]);

                    if (currImageIndex >= numGifImages) currImageIndex = 0;
                }, gifChangeSpeed);
            }).mouseout(function() {
                clearInterval(gifInterval);
            });

            // Scene dividers on scroll bar
            var allXs = []; // Scenes might have different widths? Need to store all x's
            var sceneDividerWidth = 3;
            $('.scene-change-divider-td').each(function(index) {
                var x = Math.round(($(this).offset().left / document.body.scrollWidth) * 500.0);
                //console.log('index:', index, ' x: ', x, 'x + width: ', x + sceneDividerWidth);

                allXs.push(x);

                var bar = $(document.createElementNS("http://www.w3.org/2000/svg", "rect")).attr({
                    x: x,
                    y: 0,
                    width: sceneDividerWidth,
                    height: 40,
                    fill: "red"
                });

                $scrollBar.prepend(bar);
            });

            // Scene average colours on scroll bar
            var currX = 0;
            console.log('num_scenes:', data.scenes.length, 'allx len', allXs.length);
            for (var i = 0; i < data.scenes.length; i++) {
                var r = Math.round(data.scenes[i].average_colour[0]);
                var g = Math.round(data.scenes[i].average_colour[1]);
                var b = Math.round(data.scenes[i].average_colour[2]);
                var fillColorString = "rgb(" + r + ", " + g + ", " + b + ")";


                if (i + 1 < data.scenes.length) var barWidth = allXs[i + 1] - allXs[i] - sceneDividerWidth;
                else var barWidth = allXs[i - 1 + 1] - allXs[i - 1] - sceneDividerWidth; //slighly fragile
                currX = allXs[i] + sceneDividerWidth;

                var bar = $(document.createElementNS("http://www.w3.org/2000/svg", "rect")).attr({
                    x: currX,
                    y: 0,
                    width: barWidth,
                    height: 40,
                    fill: fillColorString
                });

                $scrollBar.prepend(bar);

                //console.log('index:', i, ' curX:', currX);
                //console.log('barWidth:', barWidth);
            }

            $('#page-title').text(data.info.name);
        })
    }