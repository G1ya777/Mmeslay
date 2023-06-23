import 'package:cancellation_token_http/http.dart';
import 'package:flutter/material.dart';
import 'package:logger/logger.dart';
import 'dart:io';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:cancellation_token_http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:lottie/lottie.dart';
import 'package:permission_handler/permission_handler.dart';
import 'dart:async';
import 'package:slide_countdown/slide_countdown.dart';
import 'dialog.dart';
import 'package:mmeslay/text_editor.dart';

final Logger logger = Logger();

class Home extends StatelessWidget {
  final bool recordClicked;
  final VoidCallback setRecordClicked;

  const Home(
      {required this.recordClicked, required this.setRecordClicked, super.key});

  @override
  Widget build(BuildContext context) {
    final record = Record();

    displayNoteFn(text) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => Theme(
            data:
                Theme.of(context), // Use the same theme as the current context
            child: TextEditorWidget(initialText: text),
          ),
        ),
      );
    }

    Future<void> postAudioFile(
      File audioFile,
    ) async {
      var token = CancellationToken();
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('http://192.168.12.1:5000/upload'),
      );

      Timer timeoutTimer = Timer(const Duration(milliseconds: 7500), () {
        token.cancel();
        const snackBar = SnackBar(
          content: Text('Request timed out'),
        );
        ScaffoldMessenger.of(context).showSnackBar(snackBar);
      });

      try {
        request.files.add(
          await http.MultipartFile.fromPath(
            'audio_file',
            audioFile.path,
            contentType: MediaType('audio', 'm4a'),
          ),
        );
      } catch (e) {
        logger.d("error");
        const snackBar = SnackBar(
          content: Text('error'),
        );
        ScaffoldMessenger.of(context).showSnackBar(snackBar);
      }

      try {
        var response = await request.send(cancellationToken: token);
        timeoutTimer.cancel();
        logger.d(response.statusCode);
        if (response.statusCode == 200) {
          logger.d('Audio file uploaded successfully');
          var result = await http.Response.fromStream(response);
          logger.d(result.body);
          displayNoteFn(result.body);
        } else {
          const snackBar = SnackBar(
            content: Text('An error has occured'),
          );
          logger.d('Error uploading audio file: ${response.reasonPhrase}');
          // ignore: use_build_context_synchronously
          ScaffoldMessenger.of(context).showSnackBar(snackBar);
        }
      } catch (e) {
        const snackBar = SnackBar(
          content: Text('Server not found'),
        );
        logger.d(e);
        // ignore: use_build_context_synchronously
        ScaffoldMessenger.of(context).showSnackBar(snackBar);
      }
    }

    recordFn() async {
      if (await record.hasPermission() && !recordClicked) {
        setRecordClicked();
        Directory tempDir = await getTemporaryDirectory();
        String tempPath = tempDir.path;
        logger.d(tempPath);
        await record.start(
            path: '$tempPath/tmp.m4a',
            encoder: AudioEncoder.aacLc,
            samplingRate: 16000,
            bitRate: 64000,
            numChannels: 1);
      } else if (!await record.hasPermission()) {
        if (await Permission.microphone.isDenied) {
          // ignore: use_build_context_synchronously
          showDialog(
            context: context,
            builder: (BuildContext context) {
              return myDialog(context);
            },
          );
        }
      } else if (recordClicked) {
        setRecordClicked();
        await record.stop();
        Directory tempDir = await getTemporaryDirectory();
        String tempPath = tempDir.path;
        File audioFile = File('$tempPath/tmp.m4a');
        await postAudioFile(audioFile);
      } else {
        logger.d("undefined error");
        const snackBar = SnackBar(
          content: Text('undefined error'),
        );
        // ignore: use_build_context_synchronously
        ScaffoldMessenger.of(context).showSnackBar(snackBar);
      }
    }

    return Column(
      mainAxisAlignment: MainAxisAlignment.spaceAround,
      children: [
        recordClicked
            ? Container(
              margin: const EdgeInsets.all(2),
              child: Center(
                  child: SlideCountdown(
                    duration: const Duration(
                      seconds: 60,
                    ),
                    onDone: recordFn,
                  ),
                ),
            )
            : const Center(),
        Expanded(
          child: Container(
            padding: const EdgeInsets.only(top: 1),
            child:
                Column(mainAxisAlignment: MainAxisAlignment.center, children: [
              Center(
                child: Container(
                  child: !recordClicked
                      ? const Text(
                          'Tap to Record',
                          style: TextStyle(fontSize: 28),
                        )
                      : Container(
                          child: Lottie.asset(
                            'assets/voiceRecordAnimation.zip',
                            width: 350,
                            height: 350,
                            // fit: BoxFit.fill,
                          ),
                        ),
                ),
              ),
            ]),
          ),
        ),
        Container(
          // color: Colors.amber,
          padding: const EdgeInsets.only(bottom: 50),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              FloatingActionButton.large(
                onPressed: recordFn,
                child: !recordClicked
                    ? const Icon(Icons.keyboard_voice_outlined)
                    : const Icon(Icons.stop),
              ),
            ],
          ),
        )
      ],
    );
  }
}
