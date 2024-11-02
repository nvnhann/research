package com.udacity.pricing;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.test.context.junit4.SpringRunner;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;
import org.springframework.test.web.servlet.result.MockMvcResultMatchers;

import javax.ws.rs.core.MediaType;

@RunWith(SpringRunner.class)
@WebMvcTest
public class PricingServiceApplicationTests {

	@Autowired
	MockMvc mockMvc;

	@Test
	public void contextLoads() {
	}

	@Test
	public void testGetRequest() throws Exception {
		mockMvc.perform(MockMvcRequestBuilders
						.get("/services/price?vehicleId=1")
						.accept(MediaType.APPLICATION_JSON))
				.andExpect(MockMvcResultMatchers.status().isOk());
	}

}
