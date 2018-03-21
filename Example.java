import java.util.*;

class Example
{
	ArrayList<Double> attributes; // list of the attributes of the examples
	ArrayList<Double> target; // target class; can take 0 or 1

	/* Constructor */
	Example(ArrayList<Double> a, double t)
	{
		attributes = new ArrayList<Double>();
        target = new ArrayList<Double>();
		for(double i : a)
			attributes.add(i);
            target.add(t);
		    target.add(1-t);
	}
}
