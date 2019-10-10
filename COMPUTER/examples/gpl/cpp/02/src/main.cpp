#include <ext/stdio_filebuf.h>
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <iostream>
#include <fstream>
#include <memory>

#include <gpl/gpl.h>

#define CANVAS_WIDTH  880
#define CANVAS_HEIGHT 880
#define CANVAS_TITLE "TEST"

#define OUTER_PADDING 20
#define INNER_PADDING  1
#define BLOCK_SIZE    ((CANVAS_WIDTH - OUTER_PADDING * 2) / 28)
#define BUTTON_SIZE   (BLOCK_SIZE - INNER_PADDING * 2)

#define POINT_COLOR_BUTTON_SET   Color(0.3, 0.6, 0.9, 1.0)
#define POINT_COLOR_BUTTON_UNSET Color(0.5, 0.1, 1.0, 1.0)

std::string read(const std::string& command)
{
	FILE* process = popen(command.c_str(), "r");
	std::stringstream output;
	for (char c = getc(process); c != EOF; c = getc(process))
		output << c;

	return output.str();
}
void system(const std::string& command)
{
	system(command.c_str());
}

void set_byte(std::ofstream& file, uint16_t index)
{
	file.seekp(index);
	file << '1';
	file.seekp(0);
	file.flush();
}
void unset_byte(std::ofstream& file, uint16_t index)
{
	file.seekp(index);
	file << '0';
	file.seekp(0);
	file.flush();
}

std::istream& process_open_read(const std::string& command)
{
	int posix_handle = fileno(popen(command.c_str(), "r"));
	auto* filebuf = new __gnu_cxx::stdio_filebuf<char>(posix_handle, std::ios::in);
	return *new std::istream(filebuf);
}

pid_t g_child_pid;
void signal_to_predict(void)
{
	kill(g_child_pid, SIGUSR1);
}

int main(void)
{
	using namespace Gpl;
	using namespace Gra;
	using namespace Gra::Input::Window;
	using namespace Gra::Math;

	std::string cross_process_communication_filename = read("echo -n $(mktemp /tmp/mnist_example.XXXXXXXXX)");
	std::ofstream cross_process_communication_file(cross_process_communication_filename);
	for (uint8_t y = 0; y < 28; y++)
	for (uint8_t x = 0; x < 28; x++)
		cross_process_communication_file << '0';
	cross_process_communication_file.seekp(0);

	std::istream& prediction = process_open_read("./bin/predict.py " + cross_process_communication_filename);
	g_child_pid = std::stoi(read("ps -o pid,command | grep predict | grep -v grep | awk '{ print $1 }'"));
	std::cout << g_child_pid << std::endl;

	Canvas canvas(CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_TITLE);
	Primitive::Container button_matrix(Position(OUTER_PADDING, OUTER_PADDING), Size(canvas.width() - OUTER_PADDING * 2 , canvas.height() - OUTER_PADDING * 2));
	for (uint8_t y = 0; y < 28; y++)
	for (uint8_t x = 0; x < 28; x++)
	{
		Primitive::Shape::Point point(Position(x * BLOCK_SIZE + BLOCK_SIZE / 2, y * BLOCK_SIZE + BLOCK_SIZE / 2), POINT_COLOR_BUTTON_UNSET, BUTTON_SIZE);
		point.on_mouse_over([x, y, &canvas, &cross_process_communication_file, &prediction](Gpl::Event::Primitive::MouseOver& event)
		{
			Primitive::Shape::Point& point = event.instance<Primitive::Shape::Point>();
			if (canvas.mouse_button(Mouse::Button::LEFT) == Mouse::Action::PRESS)
			{
				if (point.color() != POINT_COLOR_BUTTON_SET)
				{
					point.color(POINT_COLOR_BUTTON_SET);
					                        set_byte(cross_process_communication_file, 27 - (27 - x + 0) + (27 - y + 0)*28);
					if (y != 27)            set_byte(cross_process_communication_file, 27 - (27 - x + 0) + (27 - y + 1)*28);
					if (x != 27)            set_byte(cross_process_communication_file, 27 - (27 - x + 1) + (27 - y + 0)*28);
					if (x != 27 && y != 27) set_byte(cross_process_communication_file, 27 - (27 - x + 1) + (27 - y + 1)*28);
					signal_to_predict();
				//	std::string line;
				//	std::getline(prediction, line);
				//	std::cout << line << std::endl;
				}
			}
			else if (canvas.mouse_button(Mouse::Button::RIGHT) == Mouse::Action::PRESS)
			{
				if (point.color() != POINT_COLOR_BUTTON_UNSET)
				{
					point.color(POINT_COLOR_BUTTON_UNSET);
					                        unset_byte(cross_process_communication_file, 27 - (27 - x + 0) + (27 - y + 0)*28);
					if (y != 27)            unset_byte(cross_process_communication_file, 27 - (27 - x + 0) + (27 - y + 1)*28);
					if (x != 27)            unset_byte(cross_process_communication_file, 27 - (27 - x + 1) + (27 - y + 0)*28);
					if (x != 27 && y != 27) unset_byte(cross_process_communication_file, 27 - (27 - x + 1) + (27 - y + 1)*28);
					signal_to_predict();
				//	std::string line;
				//	std::getline(prediction, line);
				//	std::cout << line << std::endl;
				}
			}
		});
		point.on_mouse_button([x, y, &cross_process_communication_file, &prediction](Gpl::Event::Primitive::MouseButton& event)
		{
			Primitive::Shape::Point& point = event.instance<Primitive::Shape::Point>();
			if (event.action() == Mouse::Action::PRESS)
			{
				if (event.button() == Mouse::Button::LEFT && point.color() != POINT_COLOR_BUTTON_SET)
				{
					point.color(POINT_COLOR_BUTTON_SET);
					                        set_byte(cross_process_communication_file, 27 - (27 - x + 0) + (27 - y + 0)*28);
					if (y != 27)            set_byte(cross_process_communication_file, 27 - (27 - x + 0) + (27 - y + 1)*28);
					if (x != 27)            set_byte(cross_process_communication_file, 27 - (27 - x + 1) + (27 - y + 0)*28);
					if (x != 27 && y != 27) set_byte(cross_process_communication_file, 27 - (27 - x + 1) + (27 - y + 1)*28);
					signal_to_predict();
				//	std::string line;
				//	std::getline(prediction, line);
				//	std::cout << line << std::endl;
				}
				else if (event.button() == Mouse::Button::RIGHT && point.color() != POINT_COLOR_BUTTON_UNSET)
				{
					point.color(POINT_COLOR_BUTTON_UNSET);
					                        unset_byte(cross_process_communication_file, 27 - (27 - x + 0) + (27 - y + 0)*28);
					if (y != 27)            unset_byte(cross_process_communication_file, 27 - (27 - x + 0) + (27 - y + 1)*28);
					if (x != 27)            unset_byte(cross_process_communication_file, 27 - (27 - x + 1) + (27 - y + 0)*28);
					if (x != 27 && y != 27) unset_byte(cross_process_communication_file, 27 - (27 - x + 1) + (27 - y + 1)*28);
					signal_to_predict();
				//	std::string line;
				//	std::getline(prediction, line);
				//	std::cout << line << std::endl;
				}
			}
		});
		button_matrix << std::move(point);
	}

	canvas << button_matrix;
	canvas.on_key([&](auto& event)
	{
		if (event.action() == Keyboard::Action::PRESS)
		{
			for (std::unique_ptr<Primitive::Base>& primitive : canvas.primitives<Primitive::Container>(0).primitives())
			{
				Primitive::Shape::Point& point = *(Primitive::Shape::Point*) &*primitive;
				point.color(POINT_COLOR_BUTTON_UNSET);
				cross_process_communication_file << '0';
			}
		}
		cross_process_communication_file.seekp(0);
		cross_process_communication_file.flush();
	});
	canvas.on_close([&](auto& event)
	{
		(void) event;
		system("rm " + cross_process_communication_filename);
	});

	canvas.animate();

	return 0;
}
