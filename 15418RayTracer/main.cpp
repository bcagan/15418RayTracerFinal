// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "GLFW/glfw3.h"
#include <gl/GL.h>
#include <io.h>

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
    GLFWwindow* window = glfwCreateWindow(2480, 1440, "Window Test", NULL, NULL);
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
                    "void main()\n"
                    "{\n"
                    "FragColor = vec4(ourColor,1.0f);\n"
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
    float vertices[] = {
        //positions        //colors        //UV
         0.5f,-0.5f, 0.0f, 1.0f,0.0f,0.0f, 1.f,0.f, //bottom right
        -0.5f,-0.5f, 0.0f, 0.0f,1.0f,0.0f, 0.f,0.f, //bottom left
         0.5f,-0.5f, 0.0f, 1.0f,0.0f,0.0f, 1.f,1.f, //bottom right
        -0.5f,-0.5f, 0.0f, 0.0f,1.0f,0.0f, 0.f,1.f //bottom left
    };
    
    unsigned int indices[] = {//starts from 0
        0,1,3, //tri1
        1,2,3  //tri2
    };
    
    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    
    unsigned int VBO; //The vertex buffer. Has a unique id
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO); 
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); 

    unsigned int EBO; //Element Buffer Object
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);


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


    while (!glfwWindowShouldClose(window)) {//Loop unless user closes
        //input
        processInput(window);//Run all input checking code

        glClearColor(1.0f, 0.98f, 0.3f, 1.0f); //Sets the color values to clear with
        glClear(GL_COLOR_BUFFER_BIT);//Tells opengl to clear the color buffer only, not the depth or stencil buffer

        float newVertices[] = {
            //positions        //colors        //UV
            0.5f,-0.5f, 0.0f, 1.0f,0.0f,0.0f, 1.f,0.f, //bottom right
            -0.5f,-0.5f, 0.0f, 0.0f,1.0f,0.0f, 0.f,0.f, //bottom left
            0.5f,-0.5f, 0.0f, 1.0f,0.0f,0.0f, 1.f,1.f, //bottom right
            -0.5f,-0.5f, 0.0f, 0.0f,1.0f,0.0f, 0.f,1.f //bottom left
        };

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO); 
        glBufferData(GL_ARRAY_BUFFER, sizeof(newVertices), newVertices, GL_STATIC_DRAW); 

        //Drawing a triangle
        glUseProgram(shaderProgram);//(Already set but doing anyways) set shaderprogram

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0); //Drawing the rectange from 2 triangles via EBO. Unused currently


        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        //Buffer swapping

        glfwSwapBuffers(window); //Takes the color buffer (the glfw's pixel values), and swaps
        //them to display on screen
        glfwPollEvents(); //Checks user input, updates windows state, and calls need funcs
        //That is watches for events
    } 
    
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);

    
    glfwTerminate();//Freeing function for glfw context

    return 0;
}