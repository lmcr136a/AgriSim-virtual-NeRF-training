using System;
using System.IO;
using UnityEngine;
using UnityEditor;

public static class GlobalKeyEvents
{
    // Automatically starts the Unity Editor 
    [InitializeOnLoadMethod]
    private static void EditorInit()
    {
        Debug.Log("Entering play mode..."); 
        UnityEditor.EditorApplication.EnterPlaymode();
    }
}
