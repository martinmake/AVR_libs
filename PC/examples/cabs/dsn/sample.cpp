{
using namespace Cabs::Positions;
using namespace Cabs::Colors;

history_graph.position({0.0, 0.0});
history_graph.size({0.6, 0.3});
history_graph.label("HISTORY GRAPH");
history_graph.is_bordered(true);
history_graph.is_shadowed(true);
history_graph.is_visible (true);
history_graph.data({ -2, 4 -5, -4, -3, -2, -1, -0.5, -0.4, -0.3, -0.2, -0.18, 0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, -6, 0 });
// for (float i = 5; i >= -5; i -= 0.2)
	// history_graph << i;

text_box.position({0.6 , 0.0});
text_box.size({0.4, 0.5});
text_box.label("TEXT BOX");
text_box.is_bordered(true);
text_box.is_shadowed(true);
text_box.is_visible (true);
text_box.padding({1, 2, 1, 2});
text_box.text(
		"<!BLUE_BLACK>"
		"   THIS IS SAMPLE TEXT\n"
		"       SAMPLE TEXT\n"
		"           TEXT\n"
		"<!GREEN_BLACK>"
		"TEXT\n"
		" TEXT\n"
		"  TEXT\n"
		"   TEXT\n"
		"    TEXT\n"
		"     TEEEEEEEEEEEEEEEE0123456789XT\n"
		"      TEXT\n"
		"         TEXT\n"
		"            TEXT\n"
		"               TEXT\n"
		"          TEXT\n"
		"         TEXT\n"
		"        TEXT\n"
		"       TEXT\n"
		"      TEXT"
	     );

widget2.position({0.6 , 0.5});
widget2.size({0.4, 0.1});
widget2.label("WIDGET 2");
widget2.is_bordered(true);
widget2.is_shadowed(true);
widget2.is_visible (true);

widget1.position({0.00, 0.3});
widget1.size({0.6, 0.3});
widget1.label("WIDGET 1");
widget1.is_bordered(true);
widget1.is_shadowed(true);
widget1.is_visible (true);

graph.position({0.00, 0.6});
graph.size({1.0, 0.4});
graph.label("GRAPH");
graph.is_bordered(true);
graph.is_shadowed(true);
graph.is_visible (true);

*this << history_graph;
*this << text_box;
*this << graph;
*this << widget1;
*this << widget2;
}
