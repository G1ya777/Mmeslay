import 'package:flutter/material.dart';

class MyAppBar extends StatefulWidget implements PreferredSizeWidget {
  final bool recordClicked;
  const MyAppBar({required this.recordClicked, super.key});
  @override
  State<MyAppBar> createState() => _MyAppBarState();
  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);
}

class _MyAppBarState extends State<MyAppBar> {
  @override
  Widget build(BuildContext context) {
    return AppBar(
      title: const Text('Mmeslay'),
      actions: [
        Container(
          padding: const EdgeInsets.only(right: 5),
          child: const ButtonBar(
            alignment: MainAxisAlignment.end,
            buttonPadding: EdgeInsets.only(right: 0),
          ),
        ),
      ],
    );
  }
}
