using System.Collections;
using System.Collections.Generic;
using System;
using System.Runtime.Serialization;
using UnityEngine;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

using AustinHarris.JsonRpc;
using Dummiesman;

public class ObjectSpawner : MonoBehaviour
{

    string obj_path = string.Empty;
    GameObject loaded_object;

    class Rpc : JsonRpcService {

        GameObject rpc_gameobject;

        public Rpc(GameObject rpc_gameobject) {
            this.rpc_gameobject = rpc_gameobject;
        }

        [JsonRpcMethod]
        void spawn_object(string obj_path) {
            string abs_obj_path = Application.dataPath + obj_path;
            if (!File.Exists(abs_obj_path)) {
                Debug.Log("File doesn't exist, shutting down application.");
                UnityEditor.EditorApplication.isPlaying = false;
            } else {
                Debug.Log("File exists, importing OBJ.");
                // check if there is already a loaded object
                if (rpc_gameobject != null) {
                    Debug.Log("Deleting current object...");
                    Destroy(rpc_gameobject);
                }
                rpc_gameobject = new OBJLoader().Load(abs_obj_path);
            }
            // Expand the object by a factor of 10
            rpc_gameobject.gameObject.transform.localScale = new Vector3(5, 5, 5);
        }
    }

    Rpc rpc;

    // Start is called before the first frame update
    void Start()
    {   
        rpc = new Rpc(loaded_object);   
    }


}

