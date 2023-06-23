import 'package:flutter/material.dart';
import 'package:json_theme/json_theme.dart';
import 'package:flutter/services.dart'; 
import 'dart:convert';

import 'home.dart';
import 'app_bar.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final themeStr = await rootBundle.loadString('assets/appainter_theme.json');
  final themeJson = jsonDecode(themeStr);
  final theme = ThemeDecoder.decodeThemeData(themeJson)!;

  runApp(MainApp(theme: theme));
}

class MainApp extends StatefulWidget {
  final ThemeData theme;
  const MainApp({Key? key, required this.theme}) : super(key: key);

  @override
  State<MainApp> createState() => _MainAppState();


}

class _MainAppState extends State<MainApp> {
  var _recordClicked = false;
  _setRecordClicked() {
    setState(() {
      if (_recordClicked) {
        _recordClicked = false;
      } else {
        _recordClicked = true;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: widget.theme,
      home: Scaffold(
        appBar: MyAppBar(
          recordClicked: _recordClicked,
        ),
        body: Home(
          recordClicked: _recordClicked,
          setRecordClicked: _setRecordClicked,
        ),
      ),
    );
  }
}
