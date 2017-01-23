#include "catch.hpp"
#include "../GPUPatternMining.Contract/CliquesContainer.h"


TEST_CASE("CliqueNotAlreadyProcessed", "CliquesContainerTests")
{
	CliquesContainer container;

	std::vector<unsigned short> clique1 { 1, 2, 3, 4 };
	std::vector<unsigned short> clique2 { 0, 1, 3, 4 };
	std::vector<unsigned short> clique3 { 0, 1 };
	std::vector<unsigned short> clique4 { 5, 6, 7 };
	std::vector<unsigned short> clique5 { 0, 3, 4 };

	container.insertClique(clique1);
	container.insertClique(clique2);
	container.insertClique(clique3);
	container.insertClique(clique4);

	bool cliqueAlreadyProcessed = container.checkCliqueExistence(clique5);

	REQUIRE(cliqueAlreadyProcessed == false);
}

TEST_CASE("CliqueProcessed", "CliquesContainerTests")
{
	CliquesContainer container;

	std::vector<unsigned short> clique1{ 1, 2, 3, 4 };
	std::vector<unsigned short> clique2{ 0, 1, 3, 4 };
	std::vector<unsigned short> clique3{ 0, 1 };
	std::vector<unsigned short> clique4{ 5, 6, 7 };
	std::vector<unsigned short> clique5{ 0, 3, 5 };

	container.insertClique(clique1);
	container.insertClique(clique2);
	container.insertClique(clique3);
	container.insertClique(clique4);
	container.insertClique(clique5);

	//same clique
	bool cliqueAlreadyProcessed = container.checkCliqueExistence(clique5);

	REQUIRE(cliqueAlreadyProcessed == true);
}

TEST_CASE("Smaller cliqe - subsclique already", "CliquesContainerTests")
{
	CliquesContainer container;

	std::vector<unsigned short> clique1{ 1, 2, 3, 4 };
	std::vector<unsigned short> clique2{ 0, 1, 3, 4 };
	std::vector<unsigned short> clique3{ 0, 1 };
	std::vector<unsigned short> clique4{ 5, 6, 7 };
	std::vector<unsigned short> clique5{ 1, 2, 3, 4, 666 };

	container.insertClique(clique1);
	container.insertClique(clique2);
	container.insertClique(clique3);
	container.insertClique(clique4);

	bool cliqueAlreadyProcessed = container.checkSubcliqueExistence(clique5);

	REQUIRE(cliqueAlreadyProcessed == true);
}



TEST_CASE("Smaller cliqe - any subsclique", "CliquesContainerTests")
{
	CliquesContainer container;

	std::vector<unsigned short> clique1{ 1, 2, 3, 4 };
	std::vector<unsigned short> clique2{ 0, 1, 3, 4 };
	std::vector<unsigned short> clique3{ 0, 1 };
	std::vector<unsigned short> clique4{ 5, 6, 7 };
	std::vector<unsigned short> clique5{ 1, 2, 4, 666 };

	container.insertClique(clique1);
	container.insertClique(clique2);
	container.insertClique(clique3);
	container.insertClique(clique4);

	bool cliqueAlreadyProcessed = container.checkSubcliqueExistence(clique5);

	REQUIRE(cliqueAlreadyProcessed == false);
}




TEST_CASE("Smaller cliqe - any subsclique 2", "CliquesContainerTests")
{
	CliquesContainer container;

	std::vector<unsigned short> clique1{ 1, 2, 3, 4 };
	std::vector<unsigned short> clique2{ 0, 1, 3, 4 };
	std::vector<unsigned short> clique3{ 0, 1 };
	std::vector<unsigned short> clique4{ 5, 6, 7 };
	std::vector<unsigned short> clique5{ 0, 2, 3, 666 };

	container.insertClique(clique1);
	container.insertClique(clique2);
	container.insertClique(clique3);
	container.insertClique(clique4);

	bool cliqueAlreadyProcessed = container.checkSubcliqueExistence(clique5);

	REQUIRE(cliqueAlreadyProcessed == false);
}




TEST_CASE("bigger cliqe - any subsclique", "CliquesContainerTests")
{
	CliquesContainer container;

	std::vector<unsigned short> clique1{ 1, 2, 3, 4 };
	std::vector<unsigned short> clique2{ 1, 3, 4 };
	std::vector<unsigned short> clique3{ 1, 3, 7, 666};
	std::vector<unsigned short> clique4{ 5, 6, 7 };
	std::vector<unsigned short> clique5{ 1, 3};

	container.insertClique(clique1);
	container.insertClique(clique2);
	container.insertClique(clique3);
	container.insertClique(clique4);

	bool cliqueAlreadyProcessed = container.checkSubcliqueExistence(clique5);

	REQUIRE(cliqueAlreadyProcessed == false);
}


TEST_CASE("same size", "CliquesContainerTests")
{
	CliquesContainer container;

	std::vector<unsigned short> clique1{ 1, 2, 3, 4 };
	std::vector<unsigned short> clique2{ 0, 1, 3, 4 };

	std::vector<unsigned short> clique3{ 0, 1, 3, 4 };

	container.insertClique(clique1);
	container.insertClique(clique2);

	bool cliqueAlreadyProcessed = container.checkCliqueExistence(clique3);

	REQUIRE(cliqueAlreadyProcessed == true);
}

TEST_CASE("same size not", "CliquesContainerTests")
{
	CliquesContainer container;

	std::vector<unsigned short> clique1{ 1, 2, 3, 4 };
	std::vector<unsigned short> clique2{ 0, 1, 3, 4 };

	std::vector<unsigned short> clique3{ 0, 1, 3, 5 };

	container.insertClique(clique1);
	container.insertClique(clique2);

	bool cliqueAlreadyProcessed = container.checkCliqueExistence(clique3);

	REQUIRE(cliqueAlreadyProcessed == false);
}