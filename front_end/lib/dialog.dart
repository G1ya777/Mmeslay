import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

AlertDialog myDialog(BuildContext context) {
  return AlertDialog(
    title: const Text('Microphone access denied'),
    content: const Text('Go to the settings to grant ?'),
    contentPadding: const EdgeInsets.fromLTRB(24.0, 20.0, 24.0, 0.0),
    actions: [
      Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          TextButton(
            onPressed: () {
              openAppSettings();
              Navigator.of(context).pop(); 
            },
            child: const Text('Go'),
          ),
          const SizedBox(width: 16.0),
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
            },
            child: const Text('Close')
            )
        ],
      ),
    ],
  );
}
