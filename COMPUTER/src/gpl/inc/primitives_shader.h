#ifndef _PRIMITIVES_SHADER_H_
#define _PRIMITIVES_SHADER_H_

#define VERTEX_SHADER                        \
"#version 330                            \n" \
"                                        \n" \
"layout(location = 0) in vec4 position;  \n" \
"                                        \n" \
"uniform mat4  u_mvp;                    \n" \
"uniform float u_point_size;             \n" \
"                                        \n" \
"void main()                             \n" \
"{                                       \n" \
"       gl_Position  = u_mvp * position; \n" \
"       gl_PointSize = u_point_size;     \n" \
"}                                       \n"

#define FRAGMENT_SHADER                   \
"#version 330                         \n" \
"                                     \n" \
"layout(location = 0) out vec4 color; \n" \
"                                     \n" \
"uniform vec4 u_color;                \n" \
"                                     \n" \
"void main()                          \n" \
"{                                    \n" \
"       color = u_color;              \n" \
"}                                    \n"

#endif
