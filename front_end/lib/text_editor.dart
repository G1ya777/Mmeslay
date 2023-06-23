import 'package:flutter/material.dart';

class TextEditorWidget extends StatefulWidget {
  final String initialText;

  const TextEditorWidget({Key? key, required this.initialText})
      : super(key: key);

  @override
  TextEditorWidgetState createState() => TextEditorWidgetState();
}

class TextEditorWidgetState extends State<TextEditorWidget> {
  late TextEditingController _controller;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(text: widget.initialText);
  }

  @override
  Widget build(BuildContext context) {
    final FocusNode myFocusNode = FocusNode();
    final theme = Theme.of(context);
    return MaterialApp(
      theme:theme,
      home: Scaffold(
        appBar: AppBar(
            title: const Text('Return'),
            leading: BackButton(
              onPressed: () {
                Navigator.pop(context);
              },
            )),
        body: Column(
          children: [
            Expanded(
              child: TextField(
                readOnly: true,
                controller: _controller,
                focusNode: myFocusNode,
                maxLines: null,
                // autofocus: true,
                style: const TextStyle(fontSize: 24,
                color: Colors.black),
                decoration: const InputDecoration(
                  border: InputBorder.none,
                  contentPadding: EdgeInsets.only(
                      top: 16.0, left: 16.0, right: 16.0, bottom: 32.0),
                ),
              ),
            ),

          ],
        ),
      ),
    );
  }
}
