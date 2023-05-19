using System.Collections;
using System.Collections.Generic;
using System;
using System.Runtime.Serialization;
using UnityEngine;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

public class CameraRotator : MonoBehaviour
{
    public GameObject target_object;
    private Camera cam;

    private int num_screenshots;
    public int max_screenshots;

    public float camera_speed;

    private int frames;

    private List<Vector3> positions_list;
    private string positions_file_path;
    private Vector3 curr_position;

    private List<Quaternion> quaternions_list;
    private string quaternions_file_path;
    private Quaternion curr_quaternion;

    private List<float> camera_properties;

    // write the transforms list to a file
    public void write_positions(string file_path, List<Vector3> matrices)
    {
        StreamWriter writer = new StreamWriter(file_path, true);
        for (int i = 0; i < matrices.Count; i++) {
            writer.WriteLine(matrices[i].ToString());
        }
        writer.Close();
    }

    // write the transforms list to a file
    public void write_quaternions(string file_path, List<Quaternion> matrices)
    {
        StreamWriter writer = new StreamWriter(file_path, true);
        for (int i = 0; i < matrices.Count; i++) {
            writer.WriteLine(matrices[i].ToString());
        }
        writer.Close();
    }
    

    // Function to get the intrinsic properties of the camera
    private Matrix4x4 GetIntrinsic(Camera cam)
    {
        float pixel_aspect_ratio = (float)cam.pixelWidth / (float)cam.pixelHeight;

        float alpha_u = cam.focalLength * ((float)cam.pixelWidth / cam.sensorSize.x);
        float alpha_v = cam.focalLength * pixel_aspect_ratio * ((float)cam.pixelHeight / cam.sensorSize.y);

        float u_0 = (float)cam.pixelWidth / 2;
        float v_0 = (float)cam.pixelHeight / 2;

        //IntrinsicMatrix in row major
        Matrix4x4 camIntriMatrix = new Matrix4x4(new Vector4(alpha_u, 0f, u_0, 0f),
                                                 new Vector4(0f, alpha_v, v_0, 0f),
                                                 new Vector4(0f, 0f, 1f, 0f),
                                                 new Vector4(0f, 0f, 0f, 0f));
        return camIntriMatrix;
    }


    void Start()
    {
        cam = GetComponent<Camera>();
        num_screenshots = 0;
        Application.targetFrameRate = 60;

        positions_list = new List<Vector3>();
        quaternions_list = new List<Quaternion>();
        positions_file_path = "Assets/screenshots/positions.txt";
        quaternions_file_path = "Assets/screenshots/quaternions.txt";

        // Get all the parameters of the camera, and write it to a file
        camera_properties = new List<float>(); 
        Matrix4x4 intrinsic = GetIntrinsic(cam);
        float fx = intrinsic[0, 0];
        float fy = intrinsic[1, 1];
        float cx = intrinsic[2, 0];
        float cy = intrinsic[2, 1];
        float camera_angle_x = Mathf.Atan2(Camera.main.transform.forward.y, Camera.main.transform.forward.z) * Mathf.Rad2Deg;
        float camera_angle_y = Mathf.Atan2(Camera.main.transform.forward.x, Camera.main.transform.forward.z) * Mathf.Rad2Deg;
        // We can assume all of the Unity cameras have no distortion, so we can set the distortion parameters to 0
        float k1 = 0f;
        float k2 = 0f;
        float p1 = 0f;
        float p2 = 0f;
        camera_properties.Add(camera_angle_x); // camera_angle_x
        camera_properties.Add(camera_angle_y); // camera_angle_y
        camera_properties.Add(fx); // fl_x
        camera_properties.Add(fy); // fl_y
        camera_properties.Add(k1); // k1
        camera_properties.Add(k2); // k2
        camera_properties.Add(p1); // p1
        camera_properties.Add(p2); // p2
        camera_properties.Add(cx); // cx
        camera_properties.Add(cy); // cy
        string camera_properties_file_path = "Assets/screenshots/camera_properties.txt";
        StreamWriter camera_properties_writer = new StreamWriter(camera_properties_file_path, true);
        for (int i = 0; i < camera_properties.Count; i++) {
            camera_properties_writer.WriteLine(camera_properties[i].ToString());
        }
        camera_properties_writer.Close();
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 center_point = target_object.transform.position;
        transform.LookAt(center_point); // have the camera always pointing to the object

        // get the current position and quaternion of the camera
        curr_position = Camera.main.transform.position;
        curr_quaternion = Camera.main.transform.rotation;

        // take a screenshot every 3 seconds, with the camera rotating along a single axis
        this.transform.RotateAround(center_point, new Vector3(0f, 1f, 0f), camera_speed * Time.deltaTime);
        if (frames % 180 == 0) {
            ScreenCapture.CaptureScreenshot("Assets/screenshots/pic" + num_screenshots.ToString() + ".png");
            Debug.Log("captured screenshot " + num_screenshots.ToString());
            num_screenshots++;
            // for every screenshot, add the positions and quaternion information
            positions_list.Add(curr_position);
            quaternions_list.Add(curr_quaternion);
        }
        // If we've reached the desired number of screenshots, write all the positions and quaternions to a file and exit the program
        if (num_screenshots >= max_screenshots) {
            Debug.Log("Reached max screenshots, exiting");
            // write the positions and quaternions to a file
            write_positions(positions_file_path, positions_list);
            write_quaternions(quaternions_file_path, quaternions_list);
            UnityEditor.EditorApplication.isPlaying = false;
        }
        frames++;
    }
}
