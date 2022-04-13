// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "GLFW/glfw3.h"
#include <gl/GL.h>
#include <io.h>
#include <glad/glad.h>
#include "Camera.h"
#include "Defined.h"
#include <malloc.h>
#include "Scene.h"
#include "Object.h"

void processInput(GLFWwindow* window) { //Function for all input code
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) { //If escape is pressed
        glfwSetWindowShouldClose(window, true); //Tell the window it should next close
        //(Sets a window property should close that will be checked next loop)
    }
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

int main() {

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    //Create window object
    GLFWwindow* window = glfwCreateWindow(1280, 720, "CUDA Path Tracer", NULL, NULL);
    //(creates window of size 1440p titled Window Test
    if (window == NULL) { //Null error handling
        std::cout << "Error when creating window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window); //Makes the context of our current thread window
    
    //Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Error in initializing GLAD - Managing OpenGL pointers\n";
        return -1;
    }

    glDisable(GL_DEPTH_TEST); 

    //Set viewport for opengl's rendering window
    glViewport(0, 0, 1280, 720); //Lower left corner, size
    //Handle resizing using above defined function
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);


   
   const char* vertexShaderSource = "#version 330 core\n"
            "layout (location = 0) in vec3 aPos;\n"
            "layout (location = 1) in vec3 aColor;\n"
            "layout (location = 2) in vec2 aTex;\n"
            "out vec3 ourColor;\n"
            "out vec2 ourUV;\n"
            "void main()\n"
            "{\n"
            "gl_Position = vec4(aPos,1.0);\n"
            "ourColor = aColor;\n"
            "ourUV = aTex;\n"
            "}\0";

    //Then create a shader object, again an ID to an openGL object
    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success); //Check if compilation of
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog); //Get error message since failed
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    const char* fragmentShaderSource = "#version 330 core\n"
                    "out vec4 FragColor;\n"
                    "in vec3 ourColor;\n"
                    "in vec2 ourUV;\n"
                    "uniform sampler2D inTex;\n"
                    "void main()\n"
                    "{\n"
                    //"FragColor = vec4(ourColor,1.0f);\n"
                    "FragColor = texture(inTex,ourUV);\n"
                    "}\n\0";

    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    char infoLogFrag[512];
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLogFrag);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLogFrag << std::endl;
    }

    //Creating shader program
    unsigned int shaderProgram;
    shaderProgram = glCreateProgram();//Creates shader program object
    glAttachShader(shaderProgram, vertexShader); //First attach
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);//Then actually link

    char infoLogShaderLink[512];
    glGetShaderiv(shaderProgram, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shaderProgram, 512, NULL, infoLogShaderLink);
        std::cout << "ERROR::SHADER::LINKING_FAILED\n" << infoLogShaderLink << std::endl;
    }
    glUseProgram(shaderProgram);//Sets actuve program to our new shader
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    unsigned int indices[] = {//starts from 0
        0,1,3, //tri1
        1,2,3  //tri2
    };
    
    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);


    //pos atribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    //color atribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * (sizeof(float)), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    //uv atribute
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * (sizeof(float)), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    //Creating tex

    unsigned int tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    // Wrapping
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    //Filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    int w = 1280; int h = 720;
    Camera cam;
    for (int i = 0; i < w; i++) {
        unsigned char r = (int)((float)i / 1280.f * 255.f);
        for (int j = 0; j < h; j++) {
            unsigned char b = (int)((float)j / 720.f * 255.f);
            Color3 col(r, 0, b);
            cam.image[j][i] = col;
        }
    }

    Color3* saveImage = (Color3*)malloc(sizeof(Color3)*1280*720);
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            saveImage[j * 1280 + i] = cam.image[j][i];
        }
    }

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, (void*) saveImage);
    glGenerateMipmap(GL_TEXTURE_2D);


    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUseProgram(shaderProgram);   
    glUniform1i(glGetUniformLocation(shaderProgram, "inTex"), 0);

    //Create scene and camera
    Scene sc;
    sc.background = Color3(0.f);
    //Assume camera is facing -z with up as +y as default
    Sphere sph = Sphere(Vec3(0.f,2.f,-10.f),2.f);
    sph.Mat.albedo = Color3(255, 255, 0);
    sph.Mat.emitted = Color3(0,0,0);
    sc.addObj(sph);

    int val = 255;
    float vertices[] = {
        //positions        //colors        //UV
         1.f,-1.f, 0.0f, 1.f,0.0f,0.0f, 1.f,1.f, //bottom right
        -1.f,-1.f, 0.0f, 0.0f,1.f,0.0f, 0.f,1.f, //bottom left
        -1.f,1.f, 0.0f, 0.0f,0.0f,1.f, 0.f,0.f, //top left
         1.f,-1.f, 0.0f, 1.f,0.0f,0.0f, 1.f,1.f, //bottom right
         1.f,1.f, 0.0f, 1.f,1.f,0.0f, 1.f,0.f, //top right
        -1.f,1.f, 0.0f, 0.0f,0.0f,1.f, 0.f,0.f //top left
    }; //UV is x,y, [0,1] = [left, right], [0,1] = [top,bottom]

    while (!glfwWindowShouldClose(window)) {//Loop unless user closes
        //input
        val--;
        /*if (val <= -255) val = 255;
        int tempVal = val;
        if (tempVal < 0) tempVal = -tempVal;
        float colVal = (float)tempVal / 255.f;

        //Ray trace image
        //Store image in saveImage

        */processInput(window);//Run all input checking code

        glClearColor(1.0f, 1.f, 1.f, 1.0f); //Sets the color values to clear with
        glClear(GL_COLOR_BUFFER_BIT);//Tells opengl to clear the color buffer only, not the depth or stencil buffer
        /*float vertices[] = {
            //positions        //colors        //UV
             1.f,-1.f, 0.0f, colVal,0.0f,0.0f, 1.f,1.f, //bottom right
            -1.f,-1.f, 0.0f, 0.0f,colVal,0.0f, 0.f,1.f, //bottom left
            -1.f,1.f, 0.0f, 0.0f,0.0f,colVal, 0.f,0.f, //top left
             1.f,-1.f, 0.0f, colVal,0.0f,0.0f, 1.f,1.f, //bottom right
             1.f,1.f, 0.0f, colVal,colVal,0.0f, 1.f,0.f, //top right
            -1.f,1.f, 0.0f, 0.0f,0.0f,colVal, 0.f,0.f //top left
        }; //UV is x,y, [0,1] = [left, right], [0,1] = [top,bottom]*/


        glUseProgram(shaderProgram);//(Already set but doing anyways) set shaderprogram       
        glBindVertexArray(VAO);
        unsigned int VBO; //The vertex buffer. Has a unique id
        glGenBuffers(1, &VBO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);


        //pos atribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        //color atribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * (sizeof(float)), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        //uv atribute
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * (sizeof(float)), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);
        
       // glBindBuffer(GL_ARRAY_BUFFER, VBO); 
        //glBufferData(GL_ARRAY_BUFFER, sizeof(newVertices), newVertices, GL_STATIC_DRAW); 

        //Drawing a triangle
        glDrawArrays(GL_TRIANGLES, 0, 6);

        //Buffer swapping

        glfwSwapBuffers(window); //Takes the color buffer (the glfw's pixel values), and swaps
        //them to display on screen
        glfwPollEvents(); //Checks user input, updates windows state, and calls need funcs
        //That is watches for events
        glDeleteBuffers(1, &VBO);
    } 
    
    glDeleteVertexArrays(1, &VAO);
    glDeleteProgram(shaderProgram);

    
    glfwTerminate();//Freeing function for glfw context

    return 0;
}