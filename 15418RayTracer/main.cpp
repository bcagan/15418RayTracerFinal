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
#include <chrono>

void pathtraceInit(Scene* scene);
void pathtrace(int iter); 
void pathtraceFree();

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

    

    //Create scene and camera

    int w = 1280; int h = 720;
    Scene sc;
    Scene cornellPlain;
    Scene cornellSphere;
    Scene manyBoxes;
    Scene manySpheres;
    Scene random;
    sc.background = Color3(0.f);
    cornellPlain.background = Color3(0.f);
    cornellSphere.background = Color3(0.f);
    manyBoxes.background = Color3(0.f);
    manySpheres.background = Color3(0.f);
    random.background = Color3(0.f);

    for (int i = 0; i < w; i++) {
        unsigned char r = (int)((float)i / 1280.f * 255.f);
        for (int j = 0; j < h; j++) {
            unsigned char b = (int)((float)j / 720.f * 255.f);
            Color3 col(r, 0, b);
            sc.cam.img[j * 1280 + i] = col;
            cornellPlain.cam.img[j * 1280 + i] = col;
            cornellSphere.cam.img[j * 1280 + i] = col;
            random.cam.img[j * 1280 + i] = col;
            manyBoxes.cam.img[j * 1280 + i] = col;
            manySpheres.cam.img[j * 1280 + i] = col;
        }
    }

    Color3* saveImage = (Color3*)malloc(sizeof(Color3) * 1280 * 720);
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            saveImage[j * 1280 + i] = sc.cam.img[j * 1280 + i];
        }
    }

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
    
    // CHANGE HERE TO SWAP BETWEEN NAIVE AND CUDA OUTPUT (1/2)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, (void*)saveImage);
    glGenerateMipmap(GL_TEXTURE_2D);


    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUseProgram(shaderProgram);
    glUniform1i(glGetUniformLocation(shaderProgram, "inTex"), 0);


    //Cornellboxes
    Cube cufloor = Cube(Vec3(0.f, -10.f, -5.f), 10.f);
    cufloor.Mat.albedo = Color3(255, 255, 255);
    cufloor.Mat.emitted = Color3(0, 0, 0);
    Cube culeft = Cube(Vec3(10.f, 0.f, -5.f), 10.f);
    culeft.Mat.albedo = Color3(255, 0, 0);
    culeft.Mat.emitted = Color3(0, 0, 0);
    Cube curight = Cube(Vec3(-10.f, 0.f, -5.f), 10.f);
    curight.Mat.albedo = Color3(0, 255, 0);
    curight.Mat.emitted = Color3(0, 0, 0);
    Cube curoof = Cube(Vec3(0.f, 10.f, -5.f), 10.f);
    curoof.Mat.albedo = Color3(255, 255, 255);
    curoof.Mat.emitted = Color3(255, 255, 255);
    Cube cuback = Cube(Vec3(0.f, 0.f, -10.f), 10.f);
    cuback.Mat.albedo = Color3(255, 255, 255);
    cuback.Mat.emitted = Color3(0, 0, 0);
    Sphere diffuseSphere = Sphere(Vec3(0.f, -3.f, -5.f), 3.f);
    diffuseSphere.Mat.albedo = Color3(255, 255, 255);
    diffuseSphere.Mat.emitted = Color3(0, 0, 0);

    cornellPlain.addObj(culeft);
    cornellPlain.addObj(curight);
    cornellPlain.addObj(curoof);
    cornellPlain.addObj(cufloor);
    cornellPlain.addObj(cuback);

    cornellPlain.addObjseq(&cufloor);
    cornellPlain.addObjseq(&curoof);
    cornellPlain.addObjseq(&culeft);
    cornellPlain.addObjseq(&curight);
    cornellPlain.addObjseq(&cuback);
    

    cornellSphere.addObj(culeft);
    cornellSphere.addObj(curight);
    cornellSphere.addObj(curoof);
    cornellSphere.addObj(cufloor);
    cornellSphere.addObj(cuback);
    cornellSphere.addObj(diffuseSphere);

    cornellSphere.addObjseq(&cufloor);
    cornellSphere.addObjseq(&curoof);
    cornellSphere.addObjseq(&culeft);
    cornellSphere.addObjseq(&curight);
    cornellSphere.addObjseq(&cuback);
    cornellSphere.addObjseq(&diffuseSphere);


    //Many boxes
    Cube miniCubes[13 * 13];
    for (float x = -30.f; x <= 30.f; x += 5.f) {
        for (float y = -30.f; y <= 30.f; y += 5.f) {
            Cube cubetemp = Cube(Vec3(x, y, -20.f), 2.f);
            Color3 emittedTemp = Color3(Vec3((x + 30.f) / 60.f, (y + 30.f) / 60.f, 1.f));
            cubetemp.Mat.albedo = Color3(255);
            cubetemp.Mat.emitted = Color3(emittedTemp);
            manyBoxes.addObj(cubetemp);
            manyBoxes.addObjseq(&cubetemp);
        }
    }


    //Many spheres
    Cube minSpheres[13 * 13];
    for (float x = -30.f; x <= 30.f; x += 5.f) {
        for (float y = -30.f; y <= 30.f; y += 5.f) {
            Sphere sphereTemp = Sphere(Vec3(x, y, -20.f), 2.f);
            Color3 emittedTemp = Color3(Vec3((x + 30.f) / 60.f, (y + 30.f) / 60.f, 1.f));
            sphereTemp.Mat.albedo = Color3(255);
            sphereTemp.Mat.emitted = Color3(emittedTemp);
            manySpheres.addObj(sphereTemp);
            manySpheres.addObjseq(&sphereTemp);
        }
    }

    for (int i = 0; i < 50; i++) {
        if (randf() < 0.5) {
            Sphere newSphere = Sphere(Vec3(randf() * 20.f - 10.f, randf() * 20.f - 10.f, -10.f * randf()),randf()*4.f);
            newSphere.Mat.albedo = Color3(Vec3(randf() / 2.f + 0.5f, randf() / 2.f + 0.5f, randf() / 2.f + 0.5f));
            newSphere.Mat.emitted = Color3(Vec3(randf()/2.f+0.5f, randf() / 2.f + 0.5f, randf() / 2.f + 0.5f));
            random.addObj(newSphere);
            random.addObjseq(&newSphere);
        }
        else {

            Cube newCube = Cube(Vec3(randf() * 20.f - 10.f, randf() * 20.f - 10.f, -10.f * randf()), randf() * 4.f);
            newCube.Mat.albedo = Color3(Vec3(randf() / 2.f + 0.5f, randf() / 2.f + 0.5f, randf() / 2.f + 0.5f));
            newCube.Mat.emitted = Color3(Vec3(randf() / 2.f + 0.5f, randf() / 2.f + 0.5f, randf() / 2.f + 0.5f));
            random.addObj(newCube);
            random.addObjseq(&newCube);
        }
    }


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


    auto currentTime = std::chrono::high_resolution_clock::now();

    

    while (!glfwWindowShouldClose(window)) {//Loop unless user closes

        auto previousTime = currentTime;
        currentTime = std::chrono::high_resolution_clock::now();
        float delta = std::chrono::duration< float >(currentTime - previousTime).count();

        processInput(window);//Run all input checking code

        // CHANGE HERE TO SWAP BETWEEN NAIVE AND CUDA OUTPUT (2/2)

        std::cout << delta << " is the delta\n";
       
        pathtraceInit(&random);
        //Ray trace image

        pathtrace(15);
        //sc.render();
        
        //Store image in saveImage
        for (int j = 0; j < h; j++) {
            for (int i = 0; i < w; i++) {
                saveImage[j * 1280 + i] = random.cam.img[j * 1280 + i];
                //saveImage[j * 1280 + i] = sc.cam.image[j][i];
            }
        }

        pathtraceFree();

        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, (void*)saveImage);
        glGenerateMipmap(GL_TEXTURE_2D);


        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glUseProgram(shaderProgram);
        glUniform1i(glGetUniformLocation(shaderProgram, "inTex"), 0);

        glClearColor(1.0f, 1.f, 1.f, 1.0f); //Sets the color values to clear with
        glClear(GL_COLOR_BUFFER_BIT);//Tells opengl to clear the color buffer only, not the depth or stencil buffer

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
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO); 
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); 

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
    free(saveImage);
    
    glfwTerminate();//Freeing function for glfw context

    return 0;
}